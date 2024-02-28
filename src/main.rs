#![feature(portable_simd)]
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::convert::TryInto;
use rayon::prelude::*;
use std::sync::{Mutex, Arc};
use std::env;
use std::process;
use std::simd::f32x4;
use std::slice;


#[derive(Debug)]
struct FmtChunk {
	audio_format: u16,
	num_channels: usize,
	sample_rate: u32,
	byte_rate: u32,
	block_align: u16,
	bits_per_sample: usize,
}

impl FmtChunk {
	fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
		if bytes.len() < 16 {
			return Err(io::Error::new(
				io::ErrorKind::InvalidData,
				"Insufficient data to create FmtChunk",
			));
		}
		
		Ok(FmtChunk {
			audio_format: u16::from_le_bytes([bytes[0], bytes[1]]),
			num_channels: u16::from_le_bytes([bytes[2], bytes[3]]) as usize,
			sample_rate: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
			byte_rate: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
			block_align: u16::from_le_bytes([bytes[12], bytes[13]]),
			bits_per_sample: u16::from_le_bytes([bytes[14], bytes[15]]) as usize,
		})
	}
}


#[derive(Debug, PartialEq, PartialOrd, Clone)]
struct Loop {
	start: usize,
	end: usize,
	error: f32,
}



fn compute_thread(samples: Vec<Vec<f32>>, start: usize, step: usize, min_length: usize, tail_length: usize, count: usize) -> Vec<Loop> {
	let max_len = samples[0].len() - tail_length;
	let best_loops = Arc::new(Mutex::new(Vec::new()));
	
	(start..max_len).into_par_iter().step_by(step).for_each(|this_start| {
		let mut local_best_loops: Vec<Loop> = Vec::new();
		let mut best_error = f32::MAX;
		
		for end in (this_start + min_length)..max_len {
			let error = estimate(&samples, this_start, end, tail_length);
			
			let this_loop = Loop { start: this_start, end: end, error: error };
			if error < best_error {
				local_best_loops.insert(0, this_loop);
				best_error = error;
			}
			
			if local_best_loops.len() >= count {
				local_best_loops.truncate(count);
			}
		}
		
		// Lock and update the global best loops
		let mut best_loops_guard = best_loops.lock().unwrap();
		for l in local_best_loops {
			best_loops_guard.push(l);
		}
	});
	
	let mut final_best_loops = best_loops.lock().unwrap().clone();
	final_best_loops.sort_by(|a, b| a.error.partial_cmp(&b.error).unwrap());
	final_best_loops.truncate(count);
	final_best_loops
}


fn parse_audio_data(data: Vec<u8>, num_channels: usize, bits_per_sample: usize) -> Vec<Vec<f32>> {
	let bytes_per_sample = bits_per_sample / 8;
	let num_samples = data.len() / bytes_per_sample / num_channels;
	
	if num_channels == 1 {
		println!("{} channel, {} samples", num_channels, num_samples);
	} else {
		println!("{} channels, {} samples", num_channels, num_samples);
	}
	
	let mut samples: Vec<Vec<f32>> = vec![Vec::with_capacity(num_samples); num_channels];
	
	for sample_idx in 0..num_samples {
		for channel_idx in 0..num_channels {
			let start = (sample_idx * num_channels + channel_idx) * bytes_per_sample;
			let sample_bytes = &data[start..start + bytes_per_sample];
			
			// Convert bytes to the appropriate sample type
			let sample = match bits_per_sample {
				16 => i16::from_le_bytes(sample_bytes.try_into().unwrap()) as f32,
				24 => {
					// Convert 3 bytes to i32
					let sample_i32 = i32::from_le_bytes([sample_bytes[0], sample_bytes[1], sample_bytes[2], 0]);
					sample_i32 as f32
				},
				32 => f32::from_le_bytes(sample_bytes.try_into().unwrap()),
				_ => unreachable!(),
			};
			
			samples[channel_idx].push(sample);
		}
	}
	
	samples
}


fn estimate(samples: &Vec<Vec<f32>>, start: usize, end: usize, tail_length: usize) -> f32 {
	let channels = samples.len();
	let mut error: f32 = 0.0;
	
	for ch in 0..channels {
		let sample_slice_start = unsafe { slice::from_raw_parts(samples[ch].as_ptr().add(start), tail_length) };
		let sample_slice_end = unsafe { slice::from_raw_parts(samples[ch].as_ptr().add(end), tail_length) };
		
		let mut i = 0;
		while i + 3 < tail_length {
			let a = unsafe { f32x4::from_array([
				*sample_slice_start.get_unchecked(i),
				*sample_slice_start.get_unchecked(i + 1),
				*sample_slice_start.get_unchecked(i + 2),
				*sample_slice_start.get_unchecked(i + 3),
			]) };
			let b = unsafe { f32x4::from_array([
				*sample_slice_end.get_unchecked(i),
				*sample_slice_end.get_unchecked(i + 1),
				*sample_slice_end.get_unchecked(i + 2),
				*sample_slice_end.get_unchecked(i + 3),
			]) };
			let diff = a - b;
			let squared = diff * diff;
			error += squared.to_array().iter().sum::<f32>();
			i += 4;
		}
		
		for j in i..tail_length {
			let a = unsafe { *sample_slice_start.get_unchecked(j) };
			let b = unsafe { *sample_slice_end.get_unchecked(j) };
			error += (a - b).powi(2);
		}
	}
	
	if !error.is_finite() {
		println!("Error is not finite");
	}
	error
}


fn normalize_audio(samples: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
	let mut min = f32::MAX;
	let mut max = f32::MIN;
	
	let mut normalized = vec![vec![0.0; samples[0].len()]; samples.len()];
	
	for ch in samples.iter() {
		for &sample in ch {
			if sample > max {
				max = sample;
			}
			if sample < min {
				min = sample;
			}
		}
	}
	
	let scale = f32::max(min.abs(), max.abs());
	
	for (ch, normalized_ch) in normalized.iter_mut().enumerate() {
		for (i, norm_sample) in normalized_ch.iter_mut().enumerate() {
			*norm_sample = samples[ch][i] / scale;
		}
	}
	
	normalized
}


fn main() -> io::Result<()> {
	let args: Vec<String> = env::args().collect();
	
		if args.len() != 7 { // args includes the program name, so we check for 7
			println!("Usage: Autoloop file.wav skip step minlen tail loopcount");
			process::exit(1);
		}
	
	let mut f = File::open(&args[1])?;
	let skip = args[2].parse::<usize>().expect("Error parsing skip as integer");
	let step = args[3].parse::<usize>().expect("Error parsing step as integer");
	let minlen = args[4].parse::<usize>().expect("Error parsing minlen as integer");
	let tail = args[5].parse::<usize>().expect("Error parsing tail as integer");
	
	let file_size = f.seek(SeekFrom::End(0))?;
	
	f.seek(SeekFrom::Start(0))?;
	
	let mut tag = [0; 4];
	let mut buffer = [0; 4];
	
	f.read_exact(&mut tag)?;
	
	// Compare the read bytes with the expected bytes 'RIFF'
	if &tag[..] != b"RIFF" {
		println!("First 4 characters do not match 'RIFF'");
		return Ok(());
	}
	
	f.read_exact(&mut buffer)?;
	// get file size (aka 'chunk size')
	// let file_size_from_header = u32::from_le_bytes(buffer) + 8;
	
	// 'WAVE'
	f.read_exact(&mut tag)?;
	if &tag[..] != b"WAVE" {
		println!("Missing 'WAVE' FORMAT header");
		return Ok(());
	}
	
	let mut current_position = f.seek(SeekFrom::Current(0))?;
	let mut fmt_header: Option<FmtChunk> = None;
	let mut audio_data: Option<Vec<Vec<f32>>> = None;
	
	let mut sample_rate: u32 = 0;
	
	while current_position < file_size {
		//println!("{} {} {}",current_position,file_size,current_position < file_size);
		// step through chunks
		f.read_exact(&mut tag)?;    // subchunk name
		let characters = String::from_utf8(tag.to_vec()).expect("Invalid UTF-8");
		f.read_exact(&mut buffer)?; // subchunk length
		let subchunk_length = u32::from_le_bytes(buffer);
		//println!("SUBCHUNK NAME: '{}' LENGTH: {}", characters, subchunk_length);
		// Create a Vec<u8> to store the file data
		let mut chunk_buffer = vec![0; subchunk_length as usize];
		f.read_exact(&mut chunk_buffer)?;
		
		if &tag[..] == b"fmt " {
			let mut chunk_array = [0u8; 16];
			chunk_array.copy_from_slice(&chunk_buffer[..16]);
			fmt_header = Some(FmtChunk::from_bytes(&chunk_array)?);
			//println!("{:?}", fmt_header);
		} else if &tag[..] == b"data" {
			if let Some(ref header) = fmt_header {
				//println!("Audio Format: {}", header.num_channels);
				sample_rate = header.sample_rate;
				audio_data = Some(parse_audio_data(chunk_buffer,header.num_channels,header.bits_per_sample));
			}
			//println!("{:?}", fmt_header.audio_format);
			//println!("DATA length {}", subchunk_length);
		}
		
		current_position = f.seek(SeekFrom::Current(0))?;
		//println!("position: {}/{}", current_position, file_size);
	}
	
	if let Some(ref a) = audio_data {
		//loops = loop(normalized, skip, step, minlen, tail, loopcnt, threadcnt);
		
		//                         compute_thread(samples, start, step, min_length, tail_length, count) 
		// 100000 4000 100000 1000 4
		let normalized_audio = normalize_audio(audio_data.unwrap());
		let best_loops = compute_thread(normalized_audio, skip, step, minlen, tail, 10);
		for l in best_loops {
			let length_in_secs: f32 = (l.end-l.start) as f32 / sample_rate as f32;
			println!("loop: {} - {} [{:.2} sec, error={}]", l.start, l.end, length_in_secs, l.error);
		}
		//println!("best_loops {:?}",best_loops);
		//println!("error {}",estimate(a,400,18000,16000));
		//println!("{}",a[0].len());
	}

	Ok(())
	
}
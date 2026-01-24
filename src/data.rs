// Copyright (c) 2026 Ztry8 (AslanD)
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

use std::{fs, path::Path};

pub fn list_learn_files() -> Vec<String> {
    let mut files = vec![];
    for entry in fs::read_dir(".").expect("Cannot read current directory") {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file()
            && let Some(name) = path.file_name()
            && name.to_string_lossy().starts_with("learn")
            && name.to_string_lossy().ends_with(".txt")
        {
            files.push(name.to_string_lossy().to_string());
        }
    }

    files.sort();
    files
}

pub fn read_learn_file(path: &Path) -> (Vec<[f32; 2]>, i64) {
    let content = fs::read_to_string(path).expect("Cannot read file");
    let mut data = Vec::new();
    let mut result = 0;
    for line in content.lines() {
        if line.starts_with("result") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            result = parts[1].parse().unwrap();
        } else {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let flux: f32 = parts[0].parse().unwrap();
            let time: f32 = parts[1].parse().unwrap();
            data.push([flux, time]);
        }
    }

    let flux_mean = data.iter().map(|d| d[0]).sum::<f32>() / data.len() as f32;

    let flux_std =
        (data.iter().map(|d| (d[0] - flux_mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();

    let time_mean = data.iter().map(|d| d[1]).sum::<f32>() / data.len() as f32;

    let time_std =
        (data.iter().map(|d| (d[1] - time_mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();

    let normalized: Vec<[f32; 2]> = data
        .into_iter()
        .map(|d| {
            [
                if flux_std > 0.0 {
                    (d[0] - flux_mean) / flux_std
                } else {
                    0.0
                },
                if time_std > 0.0 {
                    (d[1] - time_mean) / time_std
                } else {
                    0.0
                },
            ]
        })
        .collect();

    (normalized, result)
}

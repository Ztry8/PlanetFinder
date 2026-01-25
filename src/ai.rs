// Copyright (c) 2026 Ztry8 (AslanD)
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

use std::{io::BufRead, path::Path};
use tch::{
    Device, Tensor,
    nn::{self, OptimizerConfig, RNN},
};

const EPOCHS: usize = 2000;

fn prepare_tensors(files: &[String], seq_len: usize, device: Device) -> (Tensor, Tensor) {
    let mut xs = vec![];
    let mut ys = vec![];

    for path in files {
        let (mut data, y) = crate::data::read_learn_file(Path::new(path));
        if data.len() < seq_len {
            while data.len() < seq_len {
                data.push([0.0, 0.0]);
            }
        } else if data.len() > seq_len {
            data.truncate(seq_len);
        }

        let x_tensor = Tensor::of_slice(&data.concat())
            .view([1, seq_len as i64, 2])
            .to_device(device);

        xs.push(x_tensor);
        ys.push(y);
    }

    let xs_tensor = Tensor::cat(&xs, 0).to_device(device);
    let ys_tensor = Tensor::of_slice(&ys).to_device(device);

    (xs_tensor, ys_tensor)
}

pub fn train_model(files: &[String], seq_len: usize) -> nn::VarStore {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = &vs.root();

    let lstm = nn::lstm(root, 2, 64, Default::default());
    let linear = nn::linear(root, 64, 10, Default::default());

    let (xs, ys) = prepare_tensors(files, seq_len, device);

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    let mut best_loss = f64::INFINITY;

    println!(
        "\nStarting training on {} files for {} epochs...",
        files.len(),
        EPOCHS
    );

    for epoch in 1..=EPOCHS {
        let (output, _) = lstm.seq(&xs);
        let last_hidden = output.select(1, seq_len as i64 - 1);
        let logits = last_hidden.apply(&linear);
        let loss = logits.cross_entropy_for_logits(&ys);
        opt.backward_step(&loss);

        let loss_val = f64::from(&loss);

        if epoch % 50 == 0 || epoch == EPOCHS {
            let pct = (epoch as f64 / EPOCHS as f64) * 100.0;

            println!(
                "\nCompleted {:>4} epochs ({:.1} % done) - current error: {:.4}",
                epoch, pct, loss_val
            );

            if loss_val < best_loss {
                best_loss = loss_val;
                vs.save("model.ot").unwrap();
            }
        }
    }

    println!("\nTraining finished. Best model saved as model.ot\n");
    vs
}

pub fn predict_model(vs: &nn::VarStore, seq_len: usize) {
    let device = Device::cuda_if_available();
    let root = &vs.root();
    let lstm = nn::lstm(root, 2, 64, Default::default());
    let linear = nn::linear(root, 64, 10, Default::default());

    println!("\nEnter flux and time (like '0.998 131.2'), one per line. Type 'end' to finish:\n");
    let stdin = std::io::stdin();
    let mut data = Vec::new();

    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim() == "end" {
            break;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            println!("Invalid input, expected 'flux time'");
            continue;
        }

        let flux: f32 = parts[0].parse().unwrap();
        let time: f32 = parts[1].parse().unwrap();
        data.push([flux, time]);
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

    let mut seq_data = normalized;
    if seq_data.len() < seq_len {
        while seq_data.len() < seq_len {
            seq_data.push([0.0, 0.0]);
        }
    } else if seq_data.len() > seq_len {
        seq_data.truncate(seq_len);
    }

    let x = Tensor::of_slice(&seq_data.concat())
        .view([1, seq_len as i64, 2])
        .to_device(device);

    let (output, _) = lstm.seq(&x);
    let last_hidden = output.select(1, seq_len as i64 - 1);
    let logits = last_hidden.apply(&linear);
    let predicted = logits.argmax(-1, false);
    println!("\nPredicted number of planets: {}\n", i64::from(predicted));
}

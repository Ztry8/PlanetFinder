// Copyright (c) 2026 Ztry8 (AslanD)
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

use std::fs;
use tch::{Device, nn};

mod ai;
mod data;
mod web;

const SEQ_LEN: usize = 500;

fn usage() {
    eprintln!(
        "\nUsage:\n  planet_finder train\n  planet_finder predict\n  planet_finder web <port>\n"
    );
}

#[actix_web::main]
async fn main() {
    tch::manual_seed(42);
    
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        usage();
        return;
    }

    let files = data::list_learn_files();
    let seq_len = if files.is_empty() {
        SEQ_LEN
    } else {
        files
            .iter()
            .map(|f| fs::read_to_string(f).unwrap().lines().count() - 1)
            .max()
            .unwrap_or(SEQ_LEN)
    };

    match args[1].as_str() {
        "train" => {
            if files.is_empty() {
                eprintln!("No learn*.txt files found in current folder.");
                return;
            }
            ai::train_model(&files, seq_len);
        }
        "predict" => {
            let mut vs = nn::VarStore::new(Device::cuda_if_available());
            vs.load("model.ot").expect("Cannot load model.ot");
            ai::predict_model(&vs, seq_len);
        }
        "web" => {
            if args.len() < 3 {
                eprintln!("\nError: port required.\nUsage: planet_finder web <port>\n");
                return;
            }
            let port: u16 = args[2].parse().unwrap_or_else(|_| {
                eprintln!("Error: '{}' is not a valid port number.", args[2]);
                std::process::exit(1);
            });
            let mut vs = nn::VarStore::new(Device::cuda_if_available());
            vs.load("model.ot").expect("Cannot load model.ot");
            web::run_server(vs, seq_len, port).await.unwrap();
        }
        other => {
            eprintln!("Error: unknown command '{}'.", other);
            usage();
        }
    }
}
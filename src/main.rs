// Copyright (c) 2026 Ztry8 (AslanD)
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

use std::{fs, io};
use tch::{Device, nn};

mod ai;
mod data;

const SEQ_LEN: usize = 500;

fn main() {
    tch::manual_seed(42);

    println!("Enter 1 to train, 2 to predict:");
    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();
    let choice = choice.trim();

    let files = data::list_learn_files();
    if files.is_empty() {
        println!("No learn*.txt files found in current folder.");
        return;
    }

    let seq_len = files
        .iter()
        .map(|f| fs::read_to_string(f).unwrap().lines().count() - 1)
        .max()
        .unwrap_or(SEQ_LEN);
    //println!("Using SEQ_LEN = {}", seq_len);

    match choice {
        "1" => {
            ai::train_model(&files, seq_len);
        }
        "2" => {
            let mut vs = nn::VarStore::new(Device::cuda_if_available());
            vs.load("model.ot").expect("Cannot load model.ot");
            ai::predict_model(&vs, seq_len);
        }
        _ => println!("Invalid choice"),
    }
}

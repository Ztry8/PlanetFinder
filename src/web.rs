// Copyright (c) 2026 Ztry8 (AslanD)
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

use std::{
    collections::HashMap,
    net::IpAddr,
    sync::Mutex,
    time::{Duration, Instant},
};

use actix_web::{HttpRequest, HttpResponse, Responder, middleware, web};
use serde::{Deserialize, Serialize};
use tch::{Device, Tensor, nn::{self, RNN}};

const RATE_LIMIT: u32 = 5;
const RATE_WINDOW: Duration = Duration::from_secs(60);

pub struct RateLimiter {
    map: Mutex<HashMap<IpAddr, (u32, Instant)>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self { map: Mutex::new(HashMap::new()) }
    }

    pub fn check(&self, ip: IpAddr) -> bool {
        let mut map = self.map.lock().unwrap();
        let now = Instant::now();
        let entry = map.entry(ip).or_insert((0, now));
        if now.duration_since(entry.1) >= RATE_WINDOW {
            *entry = (1, now);
            return true;
        }
        if entry.0 >= RATE_LIMIT {
            return false;
        }
        entry.0 += 1;
        true
    }
}

#[derive(Deserialize)]
pub struct PredictRequest {
    points: Vec<Point>,
}

#[derive(Deserialize)]
struct Point {
    flux: f32,
    time: f32,
}

#[derive(Serialize)]
struct PredictResponse {
    planets: i64,
}

pub struct AppState {
    pub vs:      nn::VarStore,
    pub seq_len: usize,
    pub limiter: RateLimiter,
}

pub async fn index() -> impl Responder {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(std::fs::read_to_string("static/index.html").unwrap())
}

pub async fn predict(
    req: HttpRequest,
    state: web::Data<AppState>,
    body: web::Json<PredictRequest>,
) -> impl Responder {
    let ip = req
        .peer_addr()
        .map(|a| a.ip())
        .unwrap_or(IpAddr::from([0, 0, 0, 0]));

    if !state.limiter.check(ip) {
        return HttpResponse::TooManyRequests()
            .body("Слишком много запросов. Подождите немного.");
    }

    if body.points.is_empty() {
        return HttpResponse::BadRequest().body("Нет точек данных.");
    }

    let seq_len = state.seq_len;
    let device = Device::cuda_if_available();
    let root = &state.vs.root();
    let lstm = nn::lstm(root, 2, 64, Default::default());
    let linear = nn::linear(root, 64, 10, Default::default());

    let fluxes: Vec<f32> = body.points.iter().map(|p| p.flux).collect();
    let times:  Vec<f32> = body.points.iter().map(|p| p.time).collect();

    let f_mean = fluxes.iter().sum::<f32>() / fluxes.len() as f32;
    let f_std  = (fluxes.iter().map(|v| (v - f_mean).powi(2)).sum::<f32>() / fluxes.len() as f32).sqrt();
    let t_mean = times.iter().sum::<f32>()  / times.len()  as f32;
    let t_std  = (times.iter().map(|v| (v - t_mean).powi(2)).sum::<f32>()  / times.len()  as f32).sqrt();

    let mut seq: Vec<[f32; 2]> = body.points.iter().map(|p| [
        if f_std > 0.0 { (p.flux - f_mean) / f_std } else { 0.0 },
        if t_std > 0.0 { (p.time - t_mean) / t_std } else { 0.0 },
    ]).collect();

    while seq.len() < seq_len { seq.push([0.0, 0.0]); }
    seq.truncate(seq_len);

    let flat: Vec<f32> = seq.iter().flat_map(|p| p.iter().copied()).collect();
    let x = Tensor::of_slice(&flat)
        .view([1, seq_len as i64, 2])
        .to_device(device);

    let (output, _) = lstm.seq(&x);
    let last_hidden = output.select(1, seq_len as i64 - 1);
    let logits = last_hidden.apply(&linear);
    let planets = i64::from(logits.argmax(-1, false));

    HttpResponse::Ok().json(PredictResponse { planets })
}

pub async fn run_server(vs: nn::VarStore, seq_len: usize, port: u16) -> std::io::Result<()> {
    use actix_web::{App, HttpServer};

    println!("\nWeb server running → http://0.0.0.0:{}\n", port);

    let state = web::Data::new(AppState {
        vs,
        seq_len,
        limiter: RateLimiter::new(),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .app_data(web::JsonConfig::default().limit(1 << 20))
            .wrap(middleware::Logger::default())
            .route("/", web::get().to(index))
            .route("/predict", web::post().to(predict))
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
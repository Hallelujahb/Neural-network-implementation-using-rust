struct Neuron {
    weights: Vec<f64>,
    bias: f64,
}

struct Layer {
    neurons: Vec<Neuron>,
}

fn rand_weight(seed: f64) -> (f64, f64) {
    let next_seed = (seed * 9301.0 + 49297.0) % 233280.0;
    let value = (next_seed / 233280.0) * 2.0 - 1.0;
    (next_seed, value)
}

fn new_neuron(num_inputs: u32, start_seed: f64) -> (Neuron, f64) {
    let mut seed = start_seed;

    let weights: Vec<f64> = (0..num_inputs)
        .map(|_| {
            let (next_seed, w) = rand_weight(seed);
            seed = next_seed;
            w
        })
        .collect();

    (Neuron { weights, bias: 0.0 }, seed)
}

fn new_layer(num_inputs: u32, num_neurons: u32, start_seed: f64) -> (Layer, f64) {
    let mut seed = start_seed;

    let neurons: Vec<Neuron> = (0..num_neurons)
        .map(|_| {
            let (neuron, next_seed) = new_neuron(num_inputs, seed);
            seed = next_seed;
            neuron
        })
        .collect();

    (Layer { neurons }, seed)
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn d_sigmoid(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn forward_layer(layer: &Layer, input: &[f64]) -> Vec<f64> {
    layer.neurons.iter()
        .map(|n| sigmoid(dot_product(&n.weights, input) + n.bias))
        .collect()
}

fn backward_layer(layer: &mut Layer, input: &[f64], activation: &[f64], error: &[f64], lr: f64) -> Vec<f64> {
    let mut input_error: Vec<f64> = input.iter().map(|_| 0.0).collect();

    for n in 0..layer.neurons.len() {
        let delta = error[n] * activation[n] * (1.0 - activation[n]);

        for i in 0..input.len() {
            input_error[i] += delta * layer.neurons[n].weights[i];
            layer.neurons[n].weights[i] -= lr * delta * input[i];
        }

        layer.neurons[n].bias -= lr * delta;
    }

    input_error
}

fn main() {
    let (mut layer1, seed) = new_layer(2, 4, 42.0);
    let (mut layer2, _seed) = new_layer(4, 1, seed);

    let data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    let epochs = 10000;
    let lr = 0.5;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        for (input, target) in data {
            let hidden_activation = forward_layer(&layer1, &input);
            let final_activation = forward_layer(&layer2, &hidden_activation);

            let error: Vec<f64> = final_activation.iter()
                .zip(target.iter())
                .map(|(a, t)| a - t)
                .collect();

            total_loss += error.iter().map(|e| e * e).sum::<f64>() / error.len() as f64;

            let hidden_error = backward_layer(&mut layer2, &hidden_activation, &final_activation, &error, lr);
            backward_layer(&mut layer1, &input, &hidden_activation, &hidden_error, lr);
        }

        if epoch % 1000 == 0 {
            println!("epoch {} loss {}", epoch, total_loss / data.len() as f64);
        }
    }

    println!("done training");

    for (input, target) in data {
        let hidden_activation = forward_layer(&layer1, &input);
        let final_activation = forward_layer(&layer2, &hidden_activation);

        println!("{:?} predicted {:.3} expected {:?}", input, final_activation[0], target);
    }
}

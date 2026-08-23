# Live Training Dashboard

CTensor can stream metrics from a running training loop to a small web dashboard, so you can watch loss and accuracy curves without waiting for the run to finish or parsing console logs afterwards.

The C++ side is deliberately thin: a Python server owns the UI, and the library posts JSON to it over HTTP.

---

## Architecture

```
 training loop  ──►  log_metrics / log_scalar  ──►  HTTP POST /update  ──►  dashborad.py  ──►  browser
   (C++)              (detached std::thread)          (curl)                (localhost:8080)
```

Each log call spawns a **detached thread** that shells out to `curl`, so a slow or absent server never stalls the training loop. Failures are swallowed by design — if the dashboard is not running, training continues silently.

### Requirements

- `python3` on `PATH`, and the dashboard script `dashborad.py` in the working directory
- `curl` on `PATH`

---

## Start the Server (`start_dashboard_server`)

### Definition

```cpp
void start_dashboard_server(int port = 8080);
```

Launches the Python dashboard as a background process — `start /B` on Windows, a backgrounded process writing to `dashboard_log.txt` on Linux and macOS — and prints the URL to stdout. Check `dashboard_log.txt` if the page does not come up.

### Usage

```cpp
start_dashboard_server();       // http://localhost:8080
start_dashboard_server(9000);   // custom port
```

### Returns

`void`

---

## Log a Scalar (`log_scalar`)

### Definition

```cpp
void log_scalar(const std::string& tag, int step, double value, int port = 8080);
```

Posts a single named value at a given step — the general-purpose channel for anything you want to plot: learning rate, gradient norm, a per-batch loss, a custom metric. Series are grouped by `tag`.

### Usage

```cpp
log_scalar("train/loss", global_step, loss.read_scalar(0));
log_scalar("lr", global_step, optimizer.lr);
log_scalar("grad_norm", global_step, norm);
```

### Returns

`void` (asynchronous, fire-and-forget)

---

## Log Epoch Metrics (`log_metrics`)

### Definition

```cpp
void log_metrics(int epoch, size_t samples, double loss, double acc, int port = 8080);
```

Posts the standard end-of-epoch bundle: epoch number, number of samples seen, loss, and accuracy. `evaluate()` (see [Training](Training.md)) calls this for you, so a loop built on the provided helpers reports to the dashboard with no extra code.

### Usage

```cpp
for (int epoch = 1; epoch <= 20; ++epoch) {
    train_epoch(model, train_loader, optimizer, epoch, 100);

    set_model_mode(model, false);
    double acc = evaluate(model, test_loader);   // already calls log_metrics

    log_metrics(epoch, seen_samples, epoch_loss, acc);   // or call it yourself
}
```

### Returns

`void` (asynchronous, fire-and-forget)

---

## Full Example

```cpp
#include "core.h"
#include "neuralnet.h"

int main() {
    start_dashboard_server(8080);   // open http://localhost:8080

    MyModel model;
    auto params = model.parameters();
    optim::AdamW optimizer(params, 1e-3);

    int step = 0;
    for (int epoch = 1; epoch <= 10; ++epoch) {
        model.train();
        for (auto& batch : train_loader) {
            optimizer.zero_grad();
            Tensor loss = criterion(model(batch.data), batch.target);
            loss.backward();
            optimizer.step();

            if (step % 10 == 0) log_scalar("train/loss", step, loss.read_scalar(0));
            ++step;
        }

        model.eval();
        evaluate(model, test_loader);   // logs epoch metrics
    }
}
```

Build the standalone demo with `cmake --build build --target test_dashboard`.

---

## Notes & Limitations

- Every call forks a thread and a `curl` process. Logging every batch of a fast loop adds real overhead — log every *N* steps, as above.
- Payloads are assembled by string concatenation, with no escaping of tag names. Keep tags to plain ASCII without quotes or backslashes.
- There is no delivery guarantee, no buffering, and no retry. The dashboard is for live monitoring; use a file or a checkpoint for anything you need to keep.

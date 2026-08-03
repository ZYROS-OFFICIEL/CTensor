#include "core.h"
#include "neuralnet.h"
#include "models/gpt2.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gpt2_huggingface <path-to-gpt2-model.safetensors> [prompt_ids...]\n";
        std::cerr << "  downloads: https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors\n";
        return 1;
    }

    GPT2Config cfg = GPT2Config::gpt2_small_124M();
    GPT2Model model(cfg);

    auto report = gpt2_io::load_huggingface(argv[1], model, true);
    std::cout << "loaded checkpoint, unexpected_keys=" << report.unexpected.size() << "\n";

    std::vector<int> prompt_ids;
    if (argc > 2) {
        for (int i = 2; i < argc; ++i) prompt_ids.push_back(std::stoi(argv[i]));
    } else {
        prompt_ids = {464, 3139, 286, 4881, 318};
    }

    model.eval();
    std::vector<int> generated = model.generate(prompt_ids, 20, 0.0);

    std::cout << "generated_ids:";
    for (int id : generated) std::cout << " " << id;
    std::cout << "\n";

    return 0;
}

#include "core.h"
#include "neuralnet.h"
#include "models/gpt2.h"
#include "models/gpt2_tokenizer.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: test_gpt2_huggingface <model.safetensors> <vocab.json> <merges.txt> [prompt text]\n";
        std::cerr << "  model:  https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors\n";
        std::cerr << "  vocab:  https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json\n";
        std::cerr << "  merges: https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt\n";
        return 1;
    }

    GPT2Config cfg = GPT2Config::gpt2_small_124M();
    GPT2Model model(cfg);

    auto report = gpt2_io::load_huggingface(argv[1], model, true);
    std::cout << "loaded checkpoint, unexpected_keys=" << report.unexpected.size() << "\n";

    GPT2Tokenizer tok = GPT2Tokenizer::from_files(argv[2], argv[3]);

    std::string prompt = (argc > 4) ? argv[4] : "The capital of France is";
    std::vector<int> prompt_ids = tok.encode(prompt);

    std::cout << "prompt: \"" << prompt << "\"\n";
    std::cout << "prompt_ids:";
    for (int id : prompt_ids) std::cout << " " << id;
    std::cout << "\n";

    model.eval();
    std::vector<int> generated = model.generate(prompt_ids, 20, 0.0);
    std::string text = tok.decode(generated);

    std::cout << "generated: \"" << text << "\"\n";

    return 0;
}

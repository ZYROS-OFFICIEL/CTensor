#include "tensor1.h"

// ---------- TESTS ----------
int main() {
    try {
        std::cout << "TESTS BEGIN\n";

        // ones + print
        Tensor a = Tensor::ones({2,3});
        std::cout << "a (ones 2x3): "; Tensor::print_(a);

        // Proxy write/read
        a[0][1] = 7.0;
        double v01 = a[0][1];
        assert(v01 == 7.0);

        // astype: Float -> Int
        Tensor ai = a.astype(DType::Int32);
        std::cout << "ai (astype int): "; Tensor::print_(ai);
        assert(static_cast<int>(std::lrint(ai[0][1])) == 7);

        // to_: in-place convert to double
        a.to_(DType::Double64);
        assert(a._dtype() == DType::Double64);

        // arange + reshape
        Tensor ar = Tensor::arange(0.0, 6.0, 1.0, DType::Float32);
        Tensor ar2 = ar.reshape({2,3});
        std::cout << "ar2 (reshape 2x3): "; Tensor::print_(ar2);

        // t_ transpose (in-place)
        ar2.t_();
        std::cout << "ar2 after t_ (swap last 2 dims): "; Tensor::print_(ar2);
        // check shape swapped
        auto sh = ar2.shape();
        assert(sh.size()==2 && sh[0]==3 && sh[1]==2);

        // permute (view)
        Tensor p = ar.reshape({2,3}).permute({1,0});
        std::cout << "permute view (1,0): "; Tensor::print_(p);
        assert(p.shape().size() == 2 && p.shape()[0] == 3 && p.shape()[1] == 2);

        // select
        Tensor sel = ar.reshape({2,3}).select(0,1); // select first dim index 1 -> shape {3}
        std::cout << "select(0,1): "; Tensor::print_(sel);
        assert(sel.shape().size() == 1 && sel.shape()[0] == 3);

        // squeeze/unsqueeze/flatten
        Tensor s = Tensor::ones({1,3,1});
        Tensor sq = s.squeeze();
        std::cout << "squeeze: "; Tensor::print_(sq);
        Tensor us = sq.unsqueeze(1);
        std::cout << "unsqueeze: "; Tensor::print_(us);
        Tensor fl = us.flatten();
        std::cout << "flatten: "; Tensor::print_(fl);
        assert(fl.shape().size() == 1 && fl.shape()[0] == fl.numel());

        // pad_to_ndim
        Tensor v1 = Tensor::arange(0,3);
        Tensor padded = pad_to_ndim(v1, 2);
        std::cout << "padded: "; Tensor::print_(padded);

        // broadcast helper
        auto bsh = broadcast_batch_shape_from_vectors(std::vector<size_t>{2,1,3}, std::vector<size_t>{1,4,3});
        std::cout << "broadcasted shape: [";
        for (auto x : bsh) { std::cout << x << " "; } std::cout << "]\n";

        // pad_to_ndim correctness spot-check
        Tensor one = Tensor::ones({3});
        Tensor pad = pad_to_ndim(one, 2); // should become (1,3) or (depending chosen left-pad)
        std::cout << "pad_to_ndim(ones{3},2): "; Tensor::print_(pad);

        std::cout << "ALL TESTS PASSED\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED with exception: " << e.what() << "\n";
        return 2;
    } catch (...) {
        std::cerr << "TEST FAILED with unknown exception\n";
        return 3;
    }
}
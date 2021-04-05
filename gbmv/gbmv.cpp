#include <torch/extension.h>



torch::Tensor gbmv(
    torch::Tensor A, int lda,
    torch::Tensor x, int incx,
    torch::Tensor beta,
    torch::Tensor y, int incy
) {

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gbmv", &gbmv, "gbmv (CUDA)");
    //m.def("forward", &gbmv_forward, "LLTM forward (CUDA)");
    //m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}

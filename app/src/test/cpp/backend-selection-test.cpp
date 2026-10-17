#include "backend-selection.hpp"
#include "translation.hpp"
#include <cassert>
#include <thread>

int main() {
    // The exact spelling returned by the pinned ggml backend must select the GPU.
    assert(is_opencl_backend_name("OpenCL"));
    assert(is_opencl_backend_name("OPENCL"));
    assert(is_opencl_backend_name("opencl"));
    assert(!is_opencl_backend_name("Vulkan"));
    assert(!is_opencl_backend_name("CPU"));
    assert(!is_opencl_backend_name("OpenCL-extra"));
    assert(!is_opencl_backend_name(nullptr));

    ModelLoadControl control;
    std::thread loader([&] {
        for (int i = 0; i < 1'000; ++i)
            control.setPhase("Reading Hy-MT2 weights: " + std::to_string(i % 101) + "%");
    });
    for (int i = 0; i < 1'000; ++i) assert(!control.phaseSnapshot().empty());
    control.cancelled.store(true);
    loader.join();
    assert(control.phaseSnapshot().find("cancellation requested") != std::string::npos);
}

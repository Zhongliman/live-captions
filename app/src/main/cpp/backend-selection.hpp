#pragma once
#include <cctype>
#include <string>

// ggml's registered backend is named "OpenCL", not "OPENCL". Backend selection must
// tolerate its capitalization without accepting unrelated GPU backends.
inline bool is_opencl_backend_name(const char * name) {
    if (!name) return false;
    std::string normalized(name);
    for (auto & value : normalized)
        value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
    return normalized == "opencl";
}

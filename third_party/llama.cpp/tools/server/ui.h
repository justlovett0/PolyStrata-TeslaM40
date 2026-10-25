#pragma once

// Headless fallback for Maxwell source distributions that do not include the
// optional llama.cpp embedded Web UI subtree. The API server still compiles
// and serves inference endpoints; the asset table is intentionally empty.

#include <cstddef>
#include <string>
#include <vector>

struct llama_ui_asset {
    std::string name;
    std::string etag;
    const unsigned char * data = nullptr;
    std::size_t size = 0;
    std::string type;
};

inline const std::vector<llama_ui_asset> & llama_ui_get_assets() {
    static const std::vector<llama_ui_asset> assets;
    return assets;
}

inline bool llama_ui_use_gzip() {
    return false;
}

inline const llama_ui_asset * llama_ui_find_asset(const std::string &) {
    return nullptr;
}

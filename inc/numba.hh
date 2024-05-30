#pragma once

#include <Matrix.hh>
#include <utility>
#include <vector>

std::vector<std::pair<int, NumbaMatrix>> loadLabeledImages(char const *images_path,
                                         char const *labels_path);

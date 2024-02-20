/*
 * Copyright (c) 2020 - 2021, VinAI. All rights reserved. All information
 * information contained herein is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */

#ifndef SVM_COMMON_LOGGER_LOGGER_H
#define SVM_COMMON_LOGGER_LOGGER_H

#include <cstdlib>
#include <memory>
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
#include "spdlog/spdlog.h"

namespace common {

class Logger
{
public:
    static void Init();
    inline static std::shared_ptr<spdlog::logger> GetLogger() { return logger_; }

private:
    static std::shared_ptr<spdlog::logger> logger_;
};

}  // namespace common

#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)

#define LOG_ERROR(...)             \
    do                             \
    {                              \
        SPDLOG_ERROR(__VA_ARGS__); \
        std::exit(EXIT_FAILURE);   \
    } while (false)

#define LOG_FATAL(...)                \
    do                                \
    {                                 \
        SPDLOG_CRITICAL(__VA_ARGS__); \
        std::exit(EXIT_FAILURE);      \
    } while (false)

#define CHECK(value, msg) \
    if (!(value))         \
    {                     \
        LOG_ERROR(msg);   \
        exit(0);          \
    }

#endif  // SVM_COMMON_LOGGER_LOGGER_H
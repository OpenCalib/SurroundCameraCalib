#include "logger.h"
#include "spdlog/sinks/stdout_color_sinks.h"

using namespace common;

std::shared_ptr<spdlog::logger> Logger::logger_;

void Logger::Init()
{
    logger_ = spdlog::stdout_color_mt("SVM_PARKING");
    logger_->set_level(spdlog::level::trace);
    spdlog::set_default_logger(logger_);
    spdlog::set_pattern("%^[%T.%e][%s:%#]%$ %v");
}
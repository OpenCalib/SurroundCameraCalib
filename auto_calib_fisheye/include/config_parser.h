/*
 * Copyright (c) 2020 - 2021, VinAI. All rights reserved. All information
 * information contained herein is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */

#ifndef SVM_COMMON_UTIL_H
#define SVM_COMMON_UTIL_H

#include <fcntl.h>
#include <stdio.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "logger/logger.h"

namespace util {

inline std::vector<std::string> splitString(const std::string &src, char delimiter)
{
    std::istringstream ss(src);
    std::string token;
    std::vector<std::string> result;
    while (std::getline(ss, token, delimiter))
    {
        result.push_back(token);
    }
    return result;
}

/*
 * @brief Sets the content of the file specified by the file_name to be the
 *        ascii representation of the input protobuf.
 * @param message
 * @param file_descriptor
 * @return */
template <typename MessageType>
bool SaveProtoToASCIIFile(const MessageType &message, int file_descriptor)
{
    using google::protobuf::TextFormat;
    using google::protobuf::io::FileOutputStream;
    using google::protobuf::io::ZeroCopyOutputStream;
    if (file_descriptor < 0)
    {
        LOG_FATAL("Invalid file descriptor.");
        return false;
    }
    ZeroCopyOutputStream *output = new FileOutputStream(file_descriptor);
    bool success                 = TextFormat::Print(message, output);
    delete output;
    close(file_descriptor);
    return success;
}

/*
 * @brief Sets the content of the file specified by the file_name to be the
 *        ascii representation of the input protobuf.
 * @param message The proto to output to the specified file.
 * @param file_name The name of the target file to set the content.
 * @return If the action is successful. */
template <typename MessageType>
bool SaveProtoToASCIIFile(const MessageType &message, const std::string &file_name)
{
    int fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
    if (fd < 0)
    {
        LOG_FATAL("Unable to open file");
        return false;
    }
    return SaveProtoToASCIIFile(message, fd);
}

/*
 * @brief Parses the content of the file specified by the file_name as ascii
 *        representation of protobufs, and merges the parsed content to the
 *        proto.
 * @param file_name The name of the file to parse whose content.
 * @param message The proto to carry the parsed content in the specified file.
 * @return If the action is successful. */
template <typename MessageType>
bool LoadProtoFromASCIIFile(const std::string &file_name, MessageType *message)
{
    using google::protobuf::TextFormat;
    using google::protobuf::io::FileInputStream;
    using google::protobuf::io::ZeroCopyInputStream;
    int file_descriptor = open(file_name.c_str(), std::ios::in);
    if (file_descriptor < 0)
    {
        LOG_FATAL("Failed to open file");

        // Failed to open;
        return false;
    }
    ZeroCopyInputStream *input = new FileInputStream(file_descriptor);
    bool success               = TextFormat::Parse(input, message);
    if (!success)
    {
        LOG_FATAL("Failed to parse file");
    }
    delete input;
    close(file_descriptor);
    return success;
}

/*
 * @brief Sets the content of the file specified by the file_name to be the
 *        binary representation of the input protobuf.
 * @param message The proto to output to the specified file.
 * @param file_name The name of the target file to set the content.
 * @return If the action is successful. */
template <typename MessageType>
bool SaveProtoToBinaryFile(const MessageType &message, const std::string &file_name)
{
    std::fstream output(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
    return message.SerializeToOstream(&output);
}

/*
 * @brief Parses the content of the file specified by the file_name as binary
 *        representation of protobufs, and merges the parsed content to the
 *        proto.
 * @param file_name The name of the file to parse whose content.
 * @param message The proto to carry the parsed content in the specified file.
 * @return If the action is successful. */
template <typename MessageType>
bool LoadProtoFromBinaryFile(const std::string &file_name, MessageType *message)
{
    std::fstream input(file_name, std::ios::in | std::ios::binary);
    if (!input.good())
    {
        LOG_FATAL("Failed to open file");
        return false;
    }
    if (!message->ParseFromIstream(&input))
    {
        LOG_FATAL("Failed to parse file");
        return false;
    }
    return true;
}

/*
 * @brief Parses the content of the file specified by the file_name as a
 *        representation of protobufs, and merges the parsed content to the
 *        proto.
 * @param file_name The name of the file to parse whose content.
 * @param message The proto to carry the parsed content in the specified file.
 * @return If the action is successful. */
template <typename MessageType>
bool LoadProtoFromFile(const std::string &file_name, MessageType *message)
{
    // Try the binary parser first if it's much likely a binary proto.
    if (splitString(file_name, '.').back() == "bin")
    {
        return LoadProtoFromBinaryFile(file_name, message) ||
               LoadProtoFromASCIIFile(file_name, message);
    }
    return LoadProtoFromASCIIFile(file_name, message) ||
           LoadProtoFromBinaryFile(file_name, message);
}

}  // namespace util

#endif  // SVM_COMMON_UTIL_H

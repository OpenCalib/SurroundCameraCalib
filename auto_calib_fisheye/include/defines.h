/*
 * Copyright (c) 2020 - 2021, VinAI. All rights reserved. All information
 * information contained herein is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */

#ifndef SVM_AUTORC_DEFINES_H
#define SVM_AUTORC_DEFINES_H

using char_t     = char;
using int8_t     = signed char;
using int16_t    = signed short;
using int32_t    = signed int;
using int64_t    = signed long;
using uint8_t    = unsigned char;
using uint16_t   = unsigned short;
using uint32_t   = unsigned int;
using uint64_t   = unsigned long;
using float32_t  = float;
using float64_t  = double;
using float128_t = long double;

enum CamID : uint32_t
{
    L = 0,
    F,
    B,
    R,
    NUM_CAM
};

#endif // SVM_AUTORC_DEFINES_H

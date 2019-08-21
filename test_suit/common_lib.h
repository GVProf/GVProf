//
// Created by find on 19-7-1.
//

#ifndef CUDA_REDSHOW_COMMON_LIB
#define CUDA_REDSHOW_COMMON_LIB

#include <string>
#include <regex>

typedef unsigned char _u8;
typedef unsigned int _u32;
typedef unsigned long long _u64;


struct ThreadId {
    int bx;
    int by;
    int bz;
    int tx;
    int ty;
    int tz;

    bool operator<(const ThreadId &o) const {
        return bz < o.bz || by < o.by || bx < o.bx || tz < o.tz || ty < o.ty || tx < o.tx;
    }
    bool operator==(const ThreadId &o) const {
        return bz == o.bz && by == o.by && bx == o.bx && tz  == o.tz && ty == o.ty && tx == o.tx;
    }
};





// namespace
using std::ifstream;
using std::string;
using std::regex;


class Variable {
public:
    long long start_addr;
    int size;
    int flag;
    std::string var_name;
    int size_per_unit;
};


#endif //CUDA_REDSHOW_COMMON_LIB

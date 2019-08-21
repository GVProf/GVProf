#include <iostream>
#include "common_lib.h"
#include <map>
#include <fstream>
#include <set>
#include <regex>
#include <list>
#include <numeric>
#include <tuple>
#include <algorithm>
#include "cxxopts.hpp"

using std::set;
using std::regex;
using std::map;
using std::regex_match;
using std::smatch;
using std::cout;
using std::endl;
using std::stoi;
using std::vector;
using std::pair;
using std::distance;
using std::list;
using std::find;
using std::accumulate;
using std::tuple;
using std::max;
using std::get;
using std::find_if;
using std::make_pair;
using std::hex;
using std::dec;
using cxxopts::Options;

void init();

ThreadId transform_tid(string bid, string tid);

void read_input_file(string input_file, string target_name);

ThreadId get_max_threadId(ThreadId a, ThreadId threadid_max);

// Temporal Redundancy-Address
void get_tra_trace_map(ThreadId tid, _u64 addr);

double calc_tra_redundancy_rate(_u64 index);

// Temporal Redundancy-Value
//@todo At this time, the hpctoolkit's log doesn't have access type, so we just regard all access as read.
void get_trv_trace_map(_u64 index, _u64 pc, ThreadId tid, _u64 addr, _u64 value);

double calc_trv_redundancy_rate(_u64 line_num);

// Spatial Redundancy Address Global Memory
void get_srag_trace_map(_u64 index, _u64 pc, ThreadId tid, _u64 addr, _u64 value);

void get_srag_trace_map_test(_u64 index, _u64 pc, ThreadId tid, _u64 addr, _u64 value);

double calc_srag_redundancy_degree(_u64 index);

double calc_srag_redundancy_degree_test(_u64 index);

// Spatial Redundancy Address Shared memory
void get_sras_trace_map(_u64 index, _u64 pc, ThreadId tid, _u64 addr, _u64 value);

pair<_u64, double> calc_sras_redundancy_rate(_u64 index);

// Spatial Redundancy Value
void get_srv_trace_map(_u64 pc, ThreadId tid, _u64 addr, _u64 value);

void calc_srv_redundancy_rate(_u64 index);


// Every thread has a ordered set to save the read addresses.
map<ThreadId, list<_u64 >> tra_list;
// trace_map: {addr1 : [rd1, rd2,], }
map<_u64, vector<int >> tra_trace_map;
// Temporal Redundancy-Value dict_queue:{thread: {addr1: val1,}}
map<ThreadId, map<_u64, _u64 >> trv_map_read;
map<ThreadId, map<_u64, _u64 >> trv_map_write;
// save every dead read and write 's index of input file.
vector<_u64> dead_read_index;
vector<_u64> dead_write_index;
long long dead_read_num, dead_write_num;
//Spatial Redundancy Address Global memory
//@todo At this time, we don't know every variable's addr range, so we regard all addr as global or shared
// {pc: {tid: [(index, addr, value), ]}}
map<_u64, map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>> srag_trace_map;
//// {pc: {tid: {addr,}, ]}}
map<_u64, map<ThreadId, set<_u64>>> srag_trace_map_test;
//{pc:{value:{thread:num}}}
map<_u64, map<_u64, map<ThreadId, _u64 >>> srv_trace_map;

regex line_read_re("0x(.+?)\\|\\((.+?)\\)\\|\\((.+?)\\)\\|0x(.+?)\\|0x(.+)");
regex tid_re("(\\d+),(\\d+),(\\d+)");
// the sizes of thread block and grid
ThreadId threadid_max;

Options options("CUDA_RedShow", "A test suit for hpctoolkit santizer");

void init() {
    dead_read_num = 0;
    dead_write_num = 0;
    options.add_options()
            ("i,input", "Input trace file",cxxopts::value<std::string>());
}

ThreadId get_max_threadId(ThreadId a, ThreadId threadid_max) {
    threadid_max.bz = max(threadid_max.bz, a.bz);
    threadid_max.by = max(threadid_max.by, a.by);
    threadid_max.bx = max(threadid_max.bx, a.bx);
    threadid_max.tz = max(threadid_max.tz, a.tz);
    threadid_max.ty = max(threadid_max.ty, a.ty);
    threadid_max.tx = max(threadid_max.tx, a.tx);
    return threadid_max;
}

ThreadId transform_tid(string s_bid, string s_tid) {
// This function will transform the raw string of bid and tid to struct ThreadId
// @arg s_bid: (2,0,0): (bx,by,bz)
    ThreadId tid = {-1, -1, -1, -1, -1, -1};
    smatch sm;
    regex_match(s_bid, sm, tid_re);
    if (sm.empty()) {
        return tid;
    }
    tid.bx = stoi(sm[1], 0, 10);
    tid.by = stoi(sm[2], 0, 10);
    tid.bz = stoi(sm[3], 0, 10);
    regex_match(s_tid, sm, tid_re);
    if (sm.empty()) {
        return tid;
    }
    tid.tx = stoi(sm[1], 0, 10);
    tid.ty = stoi(sm[2], 0, 10);
    tid.tz = stoi(sm[3], 0, 10);
    return tid;
}


void get_tra_trace_map(ThreadId tid, _u64 addr) {
    map<ThreadId, list<_u64 >>::iterator tl_it;
    tl_it = tra_list.find(tid);
//  A new thread occurs. We didn't see this thread before.
    if (tl_it == tra_list.end()) {
        list<_u64> tmp;
        tmp.push_back(addr);
        tra_list.insert(pair<ThreadId, list<_u64>>(tid, tmp));
    } else {
//        This thread has his list.
//        save the reuse distance
        list<_u64>::iterator l_it;
        l_it = find(tl_it->second.begin(), tl_it->second.end(), addr);
        int tmp_rd = 0;
//        if the addr is in thread's set, calc the rd and insert the rd into trace_map
        if (l_it != tl_it->second.end()) {
            tmp_rd = distance(l_it, tl_it->second.end()) - 1;

            auto ttm_it = tra_trace_map.find(addr);
            if (ttm_it == tra_trace_map.end()) {
                vector<int> tmp_vector;
                tmp_vector.push_back(tmp_rd);
                tra_trace_map.insert(pair<_u64, vector<int>>(addr, tmp_vector));
            } else {

                ttm_it->second.push_back(tmp_rd);
            }
            //                @todo if tmp_rd == 0 and item is in list, it's useless work to reinsert the item
//            remove the pre addr from the set
            tl_it->second.remove(addr);
        }
        tl_it->second.push_back(addr);
    }
}

void get_trv_trace_map(_u64 index, _u64 pc, ThreadId tid, _u64 addr, _u64 value) {
    map<ThreadId, map<_u64, _u64 >>::iterator tmr_it;
    tmr_it = trv_map_read.find(tid);
//    The trv_map_read doesn't have the thread's record
    if (tmr_it == trv_map_read.end()) {
        map<_u64, _u64> record;
        record.insert(pair<_u64, _u64>(addr, value));
        trv_map_read.insert(pair<ThreadId, map<_u64, _u64 >>(tid, record));
    } else {
        map<_u64, _u64>::iterator m_it;
        m_it = tmr_it->second.find(addr);
//      The trv_map_read's thread record doesn't have the current addr record.
        if (m_it == tmr_it->second.end()) {
            tmr_it->second.insert(pair<_u64, _u64>(addr, value));
        } else {
//            check whether it is redundancy
            if (value == m_it->second) {
                dead_read_num++;
                dead_read_index.push_back(index);
            } else {
                m_it->second = value;
            }
        }
    }
}

/**@arg: index, if there are loops in original code, every pc will own lot of access in same thread. The index is similar to timestamp to clarify which iteration the current access in.*/
void get_srag_trace_map(_u64 index, _u64 pc, ThreadId tid, _u64 addr, _u64 value) {
//    tuple: <_u64, _u64, _u64>(index, addr, value)
    map<_u64, map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>>::iterator stm_it;
    stm_it = srag_trace_map.find(pc);
    if (stm_it == srag_trace_map.end()) {
        map<ThreadId, vector<tuple<_u64, _u64, _u64 >>> tmp = {{tid, vector<tuple<_u64, _u64, _u64 >>{
                tuple<_u64, _u64, _u64>(index, addr, value)},}};
        srag_trace_map.insert(pair<_u64, map<ThreadId, vector<tuple<_u64, _u64, _u64>>>>(pc, tmp));
    } else {
        map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>::iterator m_it;
        m_it = stm_it->second.find(tid);
//      stm has addr key but inner doesn't have tid key
        if (m_it == stm_it->second.end()) {
            vector<tuple<_u64, _u64, _u64 >> tmp_1 = {tuple<_u64, _u64, _u64>(index, addr, value)};
            stm_it->second.insert(pair<ThreadId, vector<tuple<_u64, _u64, _u64 >>>(tid, tmp_1));
        } else {
            m_it->second.emplace_back(tuple<_u64, _u64, _u64>(index, addr, value));
        }

    }

}

// For every thread, it just accesses one addr per pc?
void get_srag_trace_map_test(_u64 index, _u64 pc, ThreadId tid, _u64 addr, _u64 value) {
//    tuple: <_u64, _u64, _u64>(index, addr, value)
    map<_u64, map<ThreadId, set<_u64 >>>::iterator stm_it;
    stm_it = srag_trace_map_test.find(pc);
    if (stm_it == srag_trace_map_test.end()) {
        map<ThreadId, set<_u64 >> tmp;
        set<_u64> tmp_s;
        tmp_s.insert(addr);
        tmp.insert(pair<ThreadId, set<_u64 >>(tid, tmp_s));
        srag_trace_map_test.insert(pair<_u64, map<ThreadId, set<_u64>>>(pc, tmp));
    } else {
        map<ThreadId, set<_u64 >>::iterator m_it;
        m_it = stm_it->second.find(tid);
//      stm has addr key but inner doesn't have tid key
        if (m_it == stm_it->second.end()) {
            set<_u64> tmp_1;
            tmp_1.insert(addr);
            stm_it->second.insert(pair<ThreadId, set<_u64>>(tid, tmp_1));
        } else {
            m_it->second.insert(addr);
        }

    }

}

void get_sras_trace_map(_u64 index, _u64 pc, ThreadId tid, _u64 addr, _u64 value) {
//    tuple: <_u64, _u64, _u64>(index, addr, value)
    map<_u64, map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>>::iterator stm_it;
    stm_it = srag_trace_map.find(pc);
    if (stm_it == srag_trace_map.end()) {
        map<ThreadId, vector<tuple<_u64, _u64, _u64 >>> tmp = {{tid, vector<tuple<_u64, _u64, _u64 >>{
                tuple<_u64, _u64, _u64>(index, addr, value)},}};
        srag_trace_map.insert(pair<_u64, map<ThreadId, vector<tuple<_u64, _u64, _u64>>>>(pc, tmp));
    } else {
        map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>::iterator m_it;
        m_it = stm_it->second.find(tid);
//      stm has addr key but inner doesn't have tid key
        if (m_it == stm_it->second.end()) {
            vector<tuple<_u64, _u64, _u64 >> tmp_1 = {tuple<_u64, _u64, _u64>(index, addr, value)};
            stm_it->second.insert(pair<ThreadId, vector<tuple<_u64, _u64, _u64 >>>(tid, tmp_1));
        } else {
            m_it->second.emplace_back(tuple<_u64, _u64, _u64>(index, addr, value));
        }

    }

}


void get_srv_trace_map(_u64 pc, ThreadId tid, _u64 addr, _u64 value) {
    auto stm_it = srv_trace_map.find(pc);
    if (stm_it == srv_trace_map.end()) {
        map<_u64, map<ThreadId, _u64 >> tmp = {{value, map<ThreadId, _u64>{
                pair<ThreadId, _u64>(tid, 1)},}};
        srv_trace_map.insert(pair<_u64, map<_u64, map<ThreadId, _u64>>>(pc, tmp));
    } else {
//        m_it is map: {value: {thread:num}}
        auto m_it = stm_it->second.find(value);
        if (m_it == stm_it->second.end()) {

            map<ThreadId, _u64> tmp_1 = {pair<ThreadId, _u64>(tid, 1)};
            stm_it->second.insert(pair<_u64, map<ThreadId, _u64 >>(value, tmp_1));
        } else {
//            has the value mapping
//            update the statistics
//            auto v_it = find_if( m_it->second.begin(), m_it->second.end(),[&tid](const pair<ThreadId, _u64 >& element){ return element.first == tid;} );
            auto it = m_it->second.find(tid);
            if (it == m_it->second.end()) {
                m_it->second.insert(pair<ThreadId, _u64>(tid, 1));
            } else {
                m_it->second[tid] += 1;
            }
        }

    }

}

double calc_tra_redundancy_rate(_u64 index) {
    double tra_rate = 0;
    long long r_sum = 0;
//    How many reuse distance has recorded.
    long long r_num = 0;
    int thread_nums =
            (threadid_max.tx + 1) * (threadid_max.ty + 1) * (threadid_max.tz + 1) * (threadid_max.bx + 1) *
            (threadid_max.by + 1) * (threadid_max.bz + 1);
    map<_u64, vector<int >>::iterator ttm_it;
    for (ttm_it = tra_trace_map.begin(); ttm_it != tra_trace_map.end(); ttm_it++) {
        r_sum += accumulate(ttm_it->second.begin(), ttm_it->second.end(), 0);
        r_num += ttm_it->second.size();
    }
    tra_rate = thread_nums == 0 ? 0 : (double)r_sum / thread_nums;
    cout <<"reuse distance sum:\t"<< r_sum << endl;
    cout <<"reuse time rate:\t"<< (double)r_num / index <<endl;
//    cout << tra_rate << endl;
    return tra_rate;

}

double calc_trv_redundancy_rate(_u64 line_num) {
    if (line_num == 0) return 0;
    cout << "dead_read_num\t" << dead_read_num << endl;
    return (double) dead_read_num / line_num;
}

double calc_srag_redundancy_degree(_u64 index) {
    _u64 all_transactions = 0;
    map<_u64, map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>>::iterator stm_it;
    int bz, by, bx, tz, ty, tx;
    for (stm_it = srag_trace_map.begin(); stm_it != srag_trace_map.end(); stm_it++) {
        for (bz = 0; bz <= threadid_max.bz; bz++) {
            for (by = 0; by <= threadid_max.by; ++by) {
                for (bx = 0; bx <= threadid_max.bx; ++bx) {
                    for (tz = 0; tz <= threadid_max.tz; ++tz) {
                        for (ty = 0; ty <= threadid_max.ty; ++ty) {
                            for (tx = 0; tx <= threadid_max.tx; tx += 32) {
                                int remain_items = 0;
                                for (int tx_i = 0; tx_i < 32; ++tx_i) {
                                    ThreadId tmp_id = {bx, by, bz, tx + tx_i, ty, tz};
//                                    m_it: map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>
                                    auto m_it = stm_it->second.find(tmp_id);
                                    if (m_it != stm_it->second.end()) {
                                        remain_items = max(remain_items, (int) m_it->second.size());
                                    }
                                }
//                                if (remain_items > 1) { cout << "remain_items is " << remain_items << endl; }
                                for (int i = 0; i < remain_items; ++i) {
//                                    per warp, per pc, per iteration of a loop
                                    set<_u64> warp_unique_cache_lines;
                                    for (int tx_i = 0; tx_i < 32; ++tx_i) {
                                        ThreadId tmp_id = {bx, by, bz, tx + tx_i, ty, tz};
                                        auto m_it = stm_it->second.find(tmp_id);
                                        if (m_it != stm_it->second.end() && m_it->second.size() > i) {
//                                            the tuple has three items now. We need the second one, addr
//                                      Every cache line is 32bytes now.
                                            int x = get<1>(m_it->second[i]) >> 5;
                                            warp_unique_cache_lines.insert(get<1>(m_it->second[tx_i]) >> 5);
                                        }
                                    }
                                    all_transactions += warp_unique_cache_lines.size();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double perfect_transaction = index  / 8;
    return all_transactions / perfect_transaction;
}


double calc_srag_redundancy_degree_test(_u64 index) {
    _u64 all_transactions = 0;
    map<_u64, map<ThreadId, set<_u64>>>::iterator stm_it;
    int bz, by, bx, tz, ty, tx;
    for (stm_it = srag_trace_map_test.begin(); stm_it != srag_trace_map_test.end(); stm_it++) {
        for (bz = 0; bz <= threadid_max.bz; bz++) {
            for (by = 0; by <= threadid_max.by; ++by) {
                for (bx = 0; bx <= threadid_max.bx; ++bx) {
                    for (tz = 0; tz <= threadid_max.tz; ++tz) {
                        for (ty = 0; ty <= threadid_max.ty; ++ty) {
                            for (tx = 0; tx <= threadid_max.tx; tx += 32) {
                                int remain_items = 0;
                                for (int tx_i = 0; tx_i < 32; ++tx_i) {
                                    ThreadId tmp_id = {bx, by, bz, tx + tx_i, ty, tz};
//                                    m_it: map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>
                                    auto m_it = stm_it->second.find(tmp_id);
                                    if (m_it != stm_it->second.end()) {
                                        remain_items = max(remain_items, (int) m_it->second.size());
                                    }
                                }
                                if (remain_items > 1) { cout << "remain_items is " << remain_items << endl; }
//                                for (int i = 0; i < remain_items; ++i) {
////                                    per warp, per pc, per iteration of a loop
//                                    set<_u64> warp_unique_cache_lines;
//                                    for (int tx_i = 0; tx_i < 32; ++tx_i) {
//                                        ThreadId tmp_id = {bx, by, bz, tx + tx_i, ty, tz};
//                                        auto m_it = stm_it->second.find(tmp_id);
//                                        if (m_it != stm_it->second.end() && m_it->second.size() > tx_i) {
////                                            the tuple has three items now. We need the second one, addr
////                                      Every cache line is 32bytes now.
//                                            int x = get<1>(m_it->second[tx_i]) >> 5;
//                                            warp_unique_cache_lines.insert(get<1>(m_it->second[tx_i]) >> 5);
//                                        }
//                                    }
//                                    all_transactions += warp_unique_cache_lines.size();
//                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double perfect_transaction = index * 4.0 / 32;
    return all_transactions / perfect_transaction;
}

pair<_u64, double> calc_sras_redundancy_rate(_u64 index) {
    _u64 conflict_time = 0;
    map<_u64, map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>>::iterator stm_it;
    int bz, by, bx, tz, ty, tx;
    for (stm_it = srag_trace_map.begin(); stm_it != srag_trace_map.end(); stm_it++) {
        for (bz = 0; bz <= threadid_max.bz; bz++) {
            for (by = 0; by <= threadid_max.by; ++by) {
                for (bx = 0; bx <= threadid_max.bx; ++bx) {
                    for (tz = 0; tz <= threadid_max.tz; ++tz) {
                        for (ty = 0; ty <= threadid_max.ty; ++ty) {
                            for (tx = 0; tx <= threadid_max.tx; tx += 32) {

//                              The max number of remain items in threads' vectors. At this moment, every pc only has one access per thread. So actually, remain_items seems equal to 1.
                                int remain_items = 0;
                                for (int tx_i = 0; tx_i < 32; ++tx_i) {
                                    ThreadId tmp_id = {bx, by, bz, tx + tx_i, ty, tz};
//                                    m_it: map<ThreadId, vector<tuple<_u64, _u64, _u64 >>>
                                    auto m_it = stm_it->second.find(tmp_id);
                                    if (m_it != stm_it->second.end()) {
                                        remain_items = max(remain_items, (int) m_it->second.size());
                                    }
                                }
//                                if (remain_items > 1) { cout << "remain_items is " << remain_items << endl; }
                                for (int i = 0; i < remain_items; ++i) {
                                    int this_iterations_valid_item = 32;
//                                    per warp, per pc, per iteration of a loop
                                    set<_u64> bank_visit;
                                    for (int tx_i = 0; tx_i < 32; ++tx_i) {
                                        ThreadId tmp_id = {bx, by, bz, tx + tx_i, ty, tz};
                                        auto m_it = stm_it->second.find(tmp_id);
                                        if (m_it != stm_it->second.end() && m_it->second.size() > i) {
//                                            the tuple has three items now. We need the second one, addr
                                            bank_visit.insert(get<1>(m_it->second[i]));
                                        } else {
//                                            Not all threads in this warp works in this iteration.
                                            this_iterations_valid_item--;
                                        }
                                    }
//                                    How many conflicts
                                    conflict_time += this_iterations_valid_item - bank_visit.size();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return make_pair(conflict_time, (double) conflict_time / index);
}

void calc_srv_redundancy_rate(_u64 index) {
//    {pc:{value:{thread:num}}}
//map<_u64, map<_u64, map<ThreadId,_u64 >>> srv_trace_map;
    for (auto stm_it = srv_trace_map.begin(); stm_it != srv_trace_map.end(); stm_it++) {
        cout << "PC:\t" << hex << stm_it->first << dec << endl;
        for (auto value_it = stm_it->second.begin(); value_it != stm_it->second.end(); value_it++) {

            _u64 cur_value_access_time = 0;
//            There's no need to clarify warps?
            for (auto &tid_it : value_it->second) {
                cur_value_access_time += tid_it.second;
            }
            cout << value_it->first << ":\t" << cur_value_access_time << "\t" << (double) cur_value_access_time / index
                 << endl;

        }
    }
}


//read input file and get every line
void read_input_file(string input_file, string target_name) {
    ifstream fin(input_file.c_str());
    string line;
//    just for trv's record of every redundancy
    _u64 index = 0;
    bool in_target_kernel = false;
    while (getline(fin, line)) {
        in_target_kernel = true;
        if (in_target_kernel) {

            smatch sm;
            regex_match(line, sm, line_read_re);
            if (sm.size() == 0) {
                cout << "This line can't match the regex:\t" << line << endl;
                continue;
            }

            _u64 pc, addr, value;
            pc = stoull(sm[1], 0, 16);
            ThreadId tid = transform_tid(sm[2], sm[3]);
            if (tid.bx == -1 || tid.by == -1 || tid.bz == -1 || tid.tx == -1 || tid.ty == -1 | tid.tz == -1) {
                cout << "Can not filter threadid from " << line << endl;
            }
            threadid_max = get_max_threadId(tid, threadid_max);
            addr = stoull(sm[4], 0, 16);
            value = stoull(sm[5], 0, 16);
            get_tra_trace_map(tid, addr);
            get_trv_trace_map(index, pc, tid, addr, value);
            get_srag_trace_map(index, pc, tid, addr, value);
//            get_srag_trace_map_test(index, pc, tid, addr, value);
//            get_srv_trace_map(pc, tid, addr, value);

            index++;
        }
    }
    cout << "tra rate\t" << calc_tra_redundancy_rate(index) << endl;
    cout << "trv rate\t" << calc_trv_redundancy_rate(index) << endl;
    cout << "srag degree" << calc_srag_redundancy_degree(index) << endl;
////    cout << "srag degree" << calc_srag_redundancy_degree_test(index) << endl;
    auto ans_t = calc_sras_redundancy_rate(index);
    cout << "sras conflict times:\t" << ans_t.first << endl << "sras conflict rate:\t" << ans_t.second << endl;

//    cout << "srv rate" << endl;
//    calc_srv_redundancy_rate(index);
//    cout << endl;
}

int main(int argc, char* argv[]) {

    init();
    auto result = options.parse(argc, argv);
    read_input_file( result["input"].as<string>(),"");
    return 0;
}

#include "network.h"
#include <hls_stream.h>
#include <ap_int.h>

void conv1(hls::stream<AXI_VAL> &s0, hls::stream<packed16> &s1) {
#pragma HLS INLINE off
   bit1 linebuf[3][28];
   bit1 win[3][3];
#pragma HLS ARRAY_PARTITION variable=linebuf complete
#pragma HLS ARRAY_PARTITION variable=win complete
#pragma HLS ARRAY_PARTITION variable=trow complete
    for (int i = 0; i < batch_size; i++) {
        ap_uint<10> cnt = 0;
        packed32 indata;
        for (int ii = 0; ii < 28; ++ii) {
            ap_uint<3> target_row = trow[ii];
            for (int jj = 0; jj < 28; ++jj) {
#pragma HLS PIPELINE II=1
                if(!(cnt & 0x1f)) {
                    indata = (packed32)(s0.read().data);
                }
                bit1 inpix = indata[(cnt & 0x1f)];
                cnt += 1;
                linebuf[target_row][jj] = inpix;
                for (int k = 0; k < 3; ++k) {
#pragma HLS UNROLL
                    win[k][0] = win[k][1];
                    win[k][1] = win[k][2];
                }
                win[0][2] = (ii >= 2) ? linebuf[trow[target_row + 1]][jj] : (bit1)0;
                win[1][2] = (ii >= 1) ? linebuf[trow[target_row + 2]][jj] : (bit1)0;
                win[2][2] = inpix;

                if (ii >= 2 && jj >= 2) {
                    packed16 outw = 0;
                    for (int f = 0; f < 16; f += 4) {
                        ap_uint<4> pos_cnt0 = 0;
                        ap_uint<4> neg_cnt0 = 0;
                        ap_uint<4> pos_cnt1 = 0;
                        ap_uint<4> neg_cnt1 = 0;
                  ap_uint<4> pos_cnt2 = 0;
                  ap_uint<4> neg_cnt2 = 0;
                  ap_uint<4> pos_cnt3 = 0;
                  ap_uint<4> neg_cnt3 = 0;
                        for (int r = 0; r < 3; ++r) {
#pragma HLS UNROLL
                            for (int c = 0; c < 3; ++c) {
#pragma HLS UNROLL
                                bit1 pix = win[r][c];
                                bit1 w_bit0 = conv1w_bin[r][c][f];
                                bit1 wn_bit0 = conv1w_bin_neg[r][c][f];
                                bit1 w_bit1 = conv1w_bin[r][c][f+1];
                                bit1 wn_bit1 = conv1w_bin_neg[r][c][f+1];
                        bit1 w_bit2 = conv1w_bin[r][c][f+2];
                        bit1 wn_bit2 = conv1w_bin_neg[r][c][f+2];
                        bit1 w_bit3 = conv1w_bin[r][c][f+3];
                        bit1 wn_bit3 = conv1w_bin_neg[r][c][f+3];
                                pos_cnt0 += (pix && w_bit0);
                                neg_cnt0 += (pix && wn_bit0);
                                pos_cnt1 += (pix && w_bit1);
                                neg_cnt1 += (pix && wn_bit1);
                        pos_cnt2 += (pix && w_bit2);
                        neg_cnt2 += (pix && wn_bit2);
                        pos_cnt3 += (pix && w_bit3);
                        neg_cnt3 += (pix && wn_bit3);
                            }
                        }
                        outw[f] = (pos_cnt0 > neg_cnt0);
                        outw[f+1] = (pos_cnt1 > neg_cnt1);
                  outw[f+2] = (pos_cnt2 > neg_cnt2);
                  outw[f+3] = (pos_cnt3 > neg_cnt3);
                    }
                    s1.write(outw);
                }
            }
        }
    }
}

void conv2(hls::stream<packed16> &s1, hls::stream<packed32> &s2) {
#pragma HLS INLINE off
   packed16 linebuf[3][26];
   packed16 win[3][3];
   const ap_int<16> pct_local2[256] =  {
		    -8, -6, -6, -4, -6, -4, -4, -2, -6, -4, -4, -2, -4, -2, -2, 0,
		    -6, -4, -4, -2, -4, -2, -2, 0, -4, -2, -2, 0, -2, 0, 0, 2,
		    -6, -4, -4, -2, -4, -2, -2, 0, -4, -2, -2, 0, -2, 0, 0, 2,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -6, -4, -4, -2, -4, -2, -2, 0, -4, -2, -2, 0, -2, 0, 0, 2,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -2, 0, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 2, 4, 4, 6,
		    -6, -4, -4, -2, -4, -2, -2, 0, -4, -2, -2, 0, -2, 0, 0, 2,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -2, 0, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 2, 4, 4, 6,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -2, 0, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 2, 4, 4, 6,
		    -2, 0, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 2, 4, 4, 6,
		     0, 2, 2, 4, 2, 4, 4, 6, 2, 4, 4, 6, 4, 6, 6, 8
		};
#pragma HLS RESOURCE variable=pct_local2 core=Register
#pragma HLS ARRAY_PARTITION variable=linebuf complete
#pragma HLS ARRAY_PARTITION variable=win complete
#pragma HLS ARRAY_PARTITION variable=trow complete
#pragma HLS ARRAY_PARTITION variable=conv2w complete
    for (int i = 0; i < batch_size; i++) {
        for(int ii = 0; ii < 26; ii++) {
            ap_uint<3> target_row = trow[ii];
            for(int jj = 0; jj < 26; jj++) {
#pragma HLS PIPELINE II=1
                packed16 inpix = s1.read();
                linebuf[target_row][jj] = inpix;
                for (int k = 0; k < 3; ++k) {
#pragma HLS UNROLL
                    win[k][0] = win[k][1];
                    win[k][1] = win[k][2];
                }
                win[0][2] = linebuf[trow[target_row + 1]][jj];
                win[1][2] = linebuf[trow[target_row + 2]][jj];
                win[2][2] = inpix;
                if(ii >= 2 && jj >= 2) {
                    packed32 w = 0;
                    for(int j = 0; j < 32; j += 4) {
                        ap_int<9> sum0 = 0;
                        ap_int<9> sum1 = 0;
                  ap_int<9> sum2 = 0;
                  ap_int<9> sum3 = 0;

                        for(int k = 0; k < 3; k++) {
#pragma HLS UNROLL
                            for(int l = 0; l < 3; l++) {
#pragma HLS UNROLL
                                packed16 x0 = ~(win[k][l] ^ conv2w[j][k][l]);
                                packed16 x1 = ~(win[k][l] ^ conv2w[j+1][k][l]);
                        packed16 x2 = ~(win[k][l] ^ conv2w[j+2][k][l]);
                        packed16 x3 = ~(win[k][l] ^ conv2w[j+3][k][l]);

//                                sum0 += popcount_table[x0.range(15, 8)];
//                                sum0 += popcount_table[x0.range(7, 0)];
//                                sum1 += popcount_table[x1.range(15, 8)];
//                                sum1 += popcount_table[x1.range(7, 0)];
//                        sum2 += popcount_table[x2.range(15, 8)];
//                        sum2 += popcount_table[x2.range(7, 0)];
//                        sum3 += popcount_table[x3.range(15, 8)];
//                        sum3 += popcount_table[x3.range(7, 0)];
                        sum0 += pct_local2[x0.range(15, 8)];
                        sum0 += pct_local2[x0.range(7, 0)];
                        sum1 += pct_local2[x1.range(15, 8)];
                        sum1 += pct_local2[x1.range(7, 0)];
                        sum2 += pct_local2[x2.range(15, 8)];
                        sum2 += pct_local2[x2.range(7, 0)];
                        sum3 += pct_local2[x3.range(15, 8)];
                        sum3 += pct_local2[x3.range(7, 0)];
                            }
                        }
                        w[j] = (bit1)(sum0 > 0);
                        w[j+1] = (bit1)(sum1 > 0);
                  w[j+2] = (bit1)(sum2 > 0);
                  w[j+3] = (bit1)(sum3 > 0);
                    }
                    s2.write(w);
                }
            }
        }
    }
}

void conv3(hls::stream<packed32> &s2, hls::stream<packed32> &s3) {
   packed32 linebuf[3][24];
   packed32 win[3][3];
   const ap_int<16> pct_local[256] =  {
		    -8, -6, -6, -4, -6, -4, -4, -2, -6, -4, -4, -2, -4, -2, -2, 0,
		    -6, -4, -4, -2, -4, -2, -2, 0, -4, -2, -2, 0, -2, 0, 0, 2,
		    -6, -4, -4, -2, -4, -2, -2, 0, -4, -2, -2, 0, -2, 0, 0, 2,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -6, -4, -4, -2, -4, -2, -2, 0, -4, -2, -2, 0, -2, 0, 0, 2,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -2, 0, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 2, 4, 4, 6,
		    -6, -4, -4, -2, -4, -2, -2, 0, -4, -2, -2, 0, -2, 0, 0, 2,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -2, 0, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 2, 4, 4, 6,
		    -4, -2, -2, 0, -2, 0, 0, 2, -2, 0, 0, 2, 0, 2, 2, 4,
		    -2, 0, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 2, 4, 4, 6,
		    -2, 0, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 2, 4, 4, 6,
		     0, 2, 2, 4, 2, 4, 4, 6, 2, 4, 4, 6, 4, 6, 6, 8
		};
#pragma HLS RESOURCE variable=pct_local core=Register
#pragma HLS ARRAY_PARTITION variable=win complete
#pragma HLS ARRAY_PARTITION variable=linebuf complete
#pragma HLS ARRAY_PARTITION variable=trow complete
#pragma HLS ARRAY_PARTITION variable=conv3w complete
    for (int i = 0; i < batch_size; i++) {
        for(int ii = 0; ii < 24; ii++) {
            ap_uint<3> target_row = trow[ii];
            for(int jj = 0; jj < 24; jj++) {
#pragma HLS PIPELINE II=1
                packed32 inpix = s2.read();
                linebuf[target_row][jj] = inpix;

                for (int k = 0; k < 3; ++k) {
#pragma HLS UNROLL
                    win[k][0] = win[k][1];
                    win[k][1] = win[k][2];
                }
                win[0][2] = linebuf[trow[target_row + 1]][jj];
                win[1][2] = linebuf[trow[target_row + 2]][jj];
                win[2][2] = inpix;
                if(ii >= 2 && jj >= 2) {
                    packed32 w = 0;
                    for(int j = 0; j < 32; j += 4) {
                        ap_int<9> sum0 = 0;
                        ap_int<9> sum1 = 0;
                  ap_int<9> sum2 = 0;
                  ap_int<9> sum3 = 0;

                        for(int k = 0; k < 3; k++) {
#pragma HLS UNROLL
                            for(int l = 0; l < 3; l++) {
#pragma HLS UNROLL
                                packed32 x0 = ~(win[k][l] ^ conv3w[j][k][l]);
                                packed32 x1 = ~(win[k][l] ^ conv3w[j+1][k][l]);
                        packed32 x2 = ~(win[k][l] ^ conv3w[j+2][k][l]);
                        packed32 x3 = ~(win[k][l] ^ conv3w[j+3][k][l]);

//                                sum0 += popcount_table[x0.range(31, 24)];
//                                sum0 += popcount_table[x0.range(23, 16)];
//                                sum0 += popcount_table[x0.range(15, 8)];
//                                sum0 += popcount_table[x0.range(7, 0)];
//
//                                sum1 += popcount_table[x1.range(31, 24)];
//                                sum1 += popcount_table[x1.range(23, 16)];
//                                sum1 += popcount_table[x1.range(15, 8)];
//                                sum1 += popcount_table[x1.range(7, 0)];
//
//                        sum2 += popcount_table[x2.range(31, 24)];
//                        sum2 += popcount_table[x2.range(23, 16)];
//                        sum2 += popcount_table[x2.range(15, 8)];
//                        sum2 += popcount_table[x2.range(7, 0)];
//
//                        sum3 += popcount_table[x3.range(31, 24)];
//                        sum3 += popcount_table[x3.range(23, 16)];
//                        sum3 += popcount_table[x3.range(15, 8)];
//                        sum3 += popcount_table[x3.range(7, 0)];
                        sum0 += pct_local[x0.range(31, 24)];
                        sum0 += pct_local[x0.range(23, 16)];
                        sum0 += pct_local[x0.range(15, 8)];
                        sum0 += pct_local[x0.range(7, 0)];

                        sum1 += pct_local[x1.range(31, 24)];
                        sum1 += pct_local[x1.range(23, 16)];
                        sum1 += pct_local[x1.range(15, 8)];
                        sum1 += pct_local[x1.range(7, 0)];

                        sum2 += pct_local[x2.range(31, 24)];
                        sum2 += pct_local[x2.range(23, 16)];
                        sum2 += pct_local[x2.range(15, 8)];
                        sum2 += pct_local[x2.range(7, 0)];

                        sum3 += pct_local[x3.range(31, 24)];
                        sum3 += pct_local[x3.range(23, 16)];
                        sum3 += pct_local[x3.range(15, 8)];
                        sum3 += pct_local[x3.range(7, 0)];
                            }
                        }
                        w[j] = (bit1)(sum0 > 0);
                        w[j+1] = (bit1)(sum1 > 0);
                  w[j+2] = (bit1)(sum2 > 0);
                  w[j+3] = (bit1)(sum3 > 0);
                    }
                    s3.write(w);
                }
            }
        }
    }
}

void pool(hls::stream<packed32> &s3, hls::stream<packed32> &s4) {
#pragma HLS INLINE off
    packed32 buf3[22][22];
#pragma HLS ARRAY_PARTITION variable=buf3 cyclic factor=2 dim=1
#pragma HLS ARRAY_PARTITION variable=buf3 cyclic factor=2 dim=2
    for (int i = 0; i < batch_size; i++) {
        for  (int ii = 0; ii < 22; ii++) {
            for (int jj = 0; jj < 22; jj++) {
               buf3[ii][jj] = s3.read();
#pragma HLS PIPELINE II=1
                if(((ii & 1) == 1) && ((jj & 1) == 1)) {
                   packed32 a = buf3[ii - 1][jj - 1];
                   packed32 b = buf3[ii - 1][jj];
                   packed32 c = buf3[ii][jj - 1];
                   packed32 d = buf3[ii][jj];
                   packed32 w = ((a & b & c) | (a & b & d) | (a & c & d) | (b & c & d));
                   s4.write(w);
                }
            }
        }
    }
}

void fc(hls::stream<packed32> &s4, hls::stream<packed_out32> &s5) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=fcwT complete dim=1
    for (int i = 0; i < batch_size; i++) {
        int sum[10] = {0};
#pragma HLS ARRAY_PARTITION variable=sum complete
        for (int j = 0; j < 121; j++) {
#pragma HLS PIPELINE II=1
            packed32 w = s4.read();
            for (int k = 0; k < 10; k++) {
                ap_int<7> p = 0;
                packed32 x = ~(w ^ fcwT[k][j]);
                p += popcount_table[x.range(31, 24)];
                p += popcount_table[x.range(23, 16)];
                p += popcount_table[x.range(15, 8)];
                p += popcount_table[x.range(7, 0)];
                sum[k] += p;
            }
        }
        for (int k = 0; k < 10; k++) {
            s5.write((packed_out32)sum[k]);
        }
    }
}

void data_out(hls::stream<packed_out32> &s5, hls::stream<AXI_VAL> &m_axis_out) {
   for(int i = 0; i < outsize; i++) {
#pragma HLS PIPELINE II = 1
      AXI_VAL out_pkt;
      out_pkt.data = (unsigned int)(s5.read().to_int());
      if (i == outsize - 1)
         out_pkt.last = 1;
      else
         out_pkt.last = 0;
      out_pkt.keep = -1;
      out_pkt.id   = 0;
      out_pkt.dest = 0;
      out_pkt.user = 0;

      m_axis_out.write(out_pkt);
   }
}

void network(hls::stream<AXI_VAL> &s_axis_in, hls::stream<AXI_VAL> &m_axis_out) {
#pragma HLS INTERFACE axis port=s_axis_in
#pragma HLS INTERFACE axis port=m_axis_out
#pragma HLS INTERFACE ap_ctrl_none port=return
    hls::stream<packed16>     s1("s1");  // conv1 conv2
    hls::stream<packed32>     s2("s2");  // conv2 conv3
    hls::stream<packed32>     s3("s3");  // conv3 pool
    hls::stream<packed32>     s4("s4");  // pool  fc
    hls::stream<packed_out32> s5("s5");  // fc write_output

    #pragma HLS STREAM variable=s1 depth=1024
    #pragma HLS STREAM variable=s2 depth=1024
    #pragma HLS STREAM variable=s3 depth=1024
    #pragma HLS STREAM variable=s4 depth=512
    #pragma HLS STREAM variable=s5 depth=512

#pragma HLS DATAFLOW
    conv1 (s_axis_in, s1);
    conv2    (s1, s2);
    conv3    (s2, s3);
    pool     (s3, s4);
    fc       (s4, s5);
    data_out(s5, m_axis_out);
}

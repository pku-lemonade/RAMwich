`timescale 1ns/1ps
// SRAM CIM Calculator (bit-serial per call, synthesizable, simple logic)
// Modes:
// - 00 (INT):  对激活位为1的列，pos加1、neg减1
// - 01 (MANT): 在 INT 结果基础上乘以 MANT_FACTOR
// - 10 (EXPA): 对激活位为1的列，按每列 shift 的 (1<<shift) 做加/减
// - 11 (EXPW): 对激活位为1的列，累加该列的（带符号）权重
// 约束: 无 function、无 $signed、仅使用可综合语句

module sram_cim_calculator #(
  parameter integer XBAR_SIZE    = 128,  // 列数
  parameter integer DATA_WIDTH   = 16,   // EXPW 权重位宽（有符号）
  parameter integer SHIFT_WIDTH  = 8,    // EXPA 移位位宽（迭代*DAC分辨率的最大值位宽）
  parameter integer ACC_WIDTH    = 48,   // 累加位宽（应足够大）
  parameter integer MANT_FACTOR  = 10    // MANT 缩放
)(
  // 模式选择：2'b00=INT, 2'b01=MANT, 2'b10=EXPA, 2'b11=EXPW
  input  wire [1:0] mode,

  // 当前 bit-plane 的激活位：act_bit_vec[j]=1 表示第 j 列本次参与
  input  wire [XBAR_SIZE-1:0] act_bit_vec,

  // 该“行”的三值权重掩码（仅用于 INT/MANT/EXPA）
  input  wire [XBAR_SIZE-1:0] pos_row_bits,
  input  wire [XBAR_SIZE-1:0] neg_row_bits,

  // EXPA：每列移位量（展平，按列打包）
  input  wire [XBAR_SIZE*SHIFT_WIDTH-1:0] eaa_shift_flat,

  // EXPW：每列带符号权重（展平，按列打包）
  input  wire [XBAR_SIZE*DATA_WIDTH-1:0] ewmvm_weight_row_flat,

  // 结果（带符号累加）
  output reg  signed [ACC_WIDTH-1:0] result
);

  integer j;
  reg signed [ACC_WIDTH-1:0] acc;

  // 临时寄存器
  reg        act_bit;
  reg [SHIFT_WIDTH-1:0] shift_slice;
  integer    sh;
  reg signed [ACC_WIDTH-1:0] one_shifted;

  reg signed [DATA_WIDTH-1:0] w_sv;
  reg signed [ACC_WIDTH-1:0]  w_ext;

  always @* begin
    acc    = {ACC_WIDTH{1'b0}};
    result = {ACC_WIDTH{1'b0}};

    case (mode)
      2'b00: begin // INT：激活位=1时，pos+1，neg-1
        for (j = 0; j < XBAR_SIZE; j = j + 1) begin
          act_bit = act_bit_vec[j];
          if (act_bit) begin
            if (pos_row_bits[j]) acc = acc + {{(ACC_WIDTH-1){1'b0}}, 1'b1};
            if (neg_row_bits[j]) acc = acc - {{(ACC_WIDTH-1){1'b0}}, 1'b1};
          end
        end
        result = acc;
      end

      2'b01: begin // MANT：在 INT 基础上乘以 MANT_FACTOR
        for (j = 0; j < XBAR_SIZE; j = j + 1) begin
          act_bit = act_bit_vec[j];
          if (act_bit) begin
            if (pos_row_bits[j]) acc = acc + {{(ACC_WIDTH-1){1'b0}}, 1'b1};
            if (neg_row_bits[j]) acc = acc - {{(ACC_WIDTH-1){1'b0}}, 1'b1};
          end
        end
        result = acc * MANT_FACTOR;
      end

      2'b10: begin // EXPA：激活位=1时，按每列 shift 的 (1<<shift) 做加/减
        for (j = 0; j < XBAR_SIZE; j = j + 1) begin
          act_bit = act_bit_vec[j];
          if (act_bit) begin
            shift_slice = eaa_shift_flat[j*SHIFT_WIDTH +: SHIFT_WIDTH];
            // 夹紧位移到 [0, ACC_WIDTH-1]
            if (shift_slice > (ACC_WIDTH-1)) begin
              sh = ACC_WIDTH-1;
            end else begin
              sh = shift_slice;
            end
            one_shifted = {{(ACC_WIDTH-1){1'b0}}, 1'b1} << sh;
            if (pos_row_bits[j]) acc = acc + one_shifted;
            if (neg_row_bits[j]) acc = acc - one_shifted;
          end
        end
        result = acc;
      end

      2'b11: begin // EXPW：激活位=1时，累加该列带符号权重
        for (j = 0; j < XBAR_SIZE; j = j + 1) begin
          act_bit = act_bit_vec[j];
          if (act_bit) begin
            w_sv  = ewmvm_weight_row_flat[j*DATA_WIDTH +: DATA_WIDTH];
            // 符号扩展到 ACC_WIDTH
            w_ext = {{(ACC_WIDTH-DATA_WIDTH){w_sv[DATA_WIDTH-1]}}, w_sv};
            acc   = acc + w_ext;
          end
        end
        result = acc;
      end

      default: begin
        result = {ACC_WIDTH{1'b0}};
      end
    endcase
  end

endmodule

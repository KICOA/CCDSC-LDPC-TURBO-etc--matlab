clear; clc; close all;
%% ===================== 1. 参数设置 =====================
num_samples_per_code = 50000;    % 每种码型样本数
sample_length = 8192;           % 样本长度
snr_range = [-10, 20];          % SNR范围
train_ratio = 0.8;              % 训练集比例
output_filename = 'fec_dataset_binary8192.mat';  % 输出文件名

% 码型配置
code_configs = struct();

% 分组码
code_configs.hamming = [6,3;7,4;12,8;15,11];
code_configs.bch = [6,3;7,4;12,8;15,11];
code_configs.rs = [7,3;6,4;7,5;15,3];

% 其他码型
code_configs.polar = [32,11;32,16;32,22;32,24];
code_configs.ldpc = [2048,1024;1536,1024;1280,1024];  % 支持码率 1/2, 2/3, 4/5
code_configs.turbo = [3576,1784;   % 码率 1/2
                      5364,1784;   % 码率 1/3
                      7152,1784;   % 码率 1/4
                      10724,1784];  % 码率 1/6
code_configs.convolutional = [5, 2;
                              6, 2;
                              7, 2;
                              8, 2];  % 约束长度从 5 到 8，码率 1/2

code_types = fieldnames(code_configs);
num_code_types = length(code_types);
total_samples = num_code_types * num_samples_per_code;
ldpc_configs = code_configs.ldpc;
num_ldpc_configs = size(ldpc_configs, 1);

% 为每个LDPC配置预生成H和G矩阵
ldpc_matrices = struct('H', cell(num_ldpc_configs, 1), ...
                       'G', cell(num_ldpc_configs, 1), ...
                       'M', zeros(num_ldpc_configs, 1), ...
                       'RATE', zeros(num_ldpc_configs, 1));

fprintf('\n预生成LDPC码的H和G矩阵...\n');
for i = 1:num_ldpc_configs
    n = ldpc_configs(i, 1);
    k = ldpc_configs(i, 2);
    
    % 根据码率设置参数（与原逻辑一致）
    if n == 2048
        RATE = 1/2;
        M = 512;
    elseif n == 1536
        RATE = 2/3;
        M = 256;
    else
        RATE = 4/5;
        M = 128;
    end
    
    % 生成并存储H和G矩阵（仅执行一次）
    H = ccsdscheckmatrix2(M, RATE);
    G = ccsdsgeneratematrix2(H, M, RATE);
    
    
    ldpc_matrices(i).H = H;
    ldpc_matrices(i).G = G;
    ldpc_matrices(i).M = M;
    ldpc_matrices(i).RATE = RATE;
end


fprintf('===================== 数据集生成配置 =====================\n');
fprintf('码型数量: %d (Hamming/BCH/RS/LDPC/Turbo/Polar/Convolutional)\n', num_code_types);
fprintf('每种码型样本数: %d → 总样本数: %d\n', num_samples_per_code, total_samples);
fprintf('样本长度: %d\n', sample_length);
fprintf('SNR范围: %d~%d dB\n', snr_range(1), snr_range(2));
fprintf('训练/测试集划分: %d/%d\n', train_ratio*100, (1-train_ratio)*100);
fprintf('输出文件: %s\n', output_filename);
fprintf('数据格式: BPSK解调后的二进制序列 (0/1)\n');
fprintf('长度处理: 截断方式\n');
fprintf('=============================================================\n');

%% ===================== 2. 初始化存储数组 =====================
X = zeros(total_samples, sample_length, 'uint8');  % 改为uint8存储0/1
y = zeros(total_samples, 1, 'int8');
snr_values = zeros(total_samples, 1, 'single');
code_params = cell(total_samples, 1);

%% ===================== 3. 主生成循环 =====================
sample_count = 0;
for code_idx = 1:num_code_types
    code_type = code_types{code_idx};
    fprintf('\n生成 %s 码数据...\n', code_type);
    
    configs = code_configs.(code_type);
    num_configs = size(configs, 1);
    
    for sample_idx = 1:num_samples_per_code
        sample_count = sample_count + 1;
        
        config_idx = randi(num_configs);
        config = configs(config_idx, :);
        
        % 使用截断方式：生成足够长的序列然后截断
        n = config(1);
        num_frames = ceil(sample_length * 1.2 / n);  % 生成稍多的帧用于截断
        
        if strcmp(code_type, 'ldpc')
            % 找到匹配的预生成矩阵索引
            ldpc_idx = find(ldpc_configs(:,1) == config(1) & ldpc_configs(:,2) == config(2));
            matrices = ldpc_matrices(ldpc_idx);
    
    % 调用生成函数时传入预生成的矩阵
            [binary_sequence, snr_db, param_info] = generate_sample(code_type, config, num_frames, sample_length, snr_range, matrices);
        else
    % 其他码型不变
    [binary_sequence, snr_db, param_info] = generate_sample(code_type, config, num_frames, sample_length, snr_range);
        end

        X(sample_count, :) = binary_sequence;
        y(sample_count) = int8(code_idx - 1);
        snr_values(sample_count) = single(snr_db);
        code_params{sample_count} = param_info;
        
        if mod(sample_idx, 100) == 0
            fprintf('  %s 码进度: %d/%d\n', code_type, sample_idx, num_samples_per_code);
        end
    end
end


%% ===================== 5. 保存数据集 =====================
fprintf('\n保存数据集到 %s...\n', output_filename);

metadata = struct();
metadata.paper_reference = 'Fine-grained recognition of error correcting codes based on 1-D CNN (2020)';
metadata.code_types = code_types;
metadata.sample_length = sample_length;
metadata.snr_range = snr_range;
metadata.num_samples_per_code = num_samples_per_code;
metadata.train_test_split = [train_ratio, 1-train_ratio];
metadata.data_format = 'BPSK解调后的二进制序列 (0/1)，使用uint8存储';
metadata.length_handling = '截断方式';
metadata.creation_date = datestr(now);
metadata.implementation_note = '所有码型均经过BPSK调制、加高斯白噪声、BPSK硬判决解调';

output_folder = fileparts(output_filename);
if ~exist(output_folder, 'dir') && ~isempty(output_folder)
    mkdir(output_folder);
end

save(output_filename, ...
    'X', 'y', 'snr_values', 'code_params', ...
    'metadata', '-v7.3');

fprintf('\n数据集生成完成！总样本数: %d\n', total_samples);
for i = 1:num_code_types
    count = sum(y == (i-1));
    fprintf('  %s: %d个样本\n', code_types{i}, count);
end

%% ===================== 6. 样本生成函数 =====================
function [binary_sequence, snr_db, param_info] = generate_sample(code_type, config, num_frames, target_length, snr_range, matrices)
    % 根据码型生成样本数据，包含BPSK解调后的二进制序列和SNR信息
    binary_sequence = [];
    snr_db = 0;
    param_info = struct();
    
    snr_db = randi(snr_range);  % 随机选择SNR值
    
    % 根据码型调用对应的生成函数
    if strcmp(code_type, 'hamming')
        binary_sequence = generate_hamming(config, num_frames, target_length, snr_db);
    elseif strcmp(code_type, 'bch')
        binary_sequence = generate_bch(config, num_frames, target_length, snr_db);
    elseif strcmp(code_type, 'rs')
        binary_sequence = generate_rs(config, num_frames, target_length, snr_db);
    elseif strcmp(code_type, 'polar')
        binary_sequence = generate_polar(config, num_frames, target_length, snr_db);
    elseif strcmp(code_type, 'ldpc')
        binary_sequence = generate_ldpc(config, num_frames, target_length, snr_db, matrices);
    elseif strcmp(code_type, 'turbo')
        binary_sequence = generate_turbo(config, num_frames, target_length, snr_db);
    elseif strcmp(code_type, 'convolutional')
        binary_sequence = generate_convolutional(config, target_length, snr_db);
    else
        error('Unsupported code type');
    end
    
    % 安全检查：确保生成的序列长度正确
    if length(binary_sequence) ~= target_length
        error('生成的二进制序列长度为 %d，与目标长度 %d 不符。', length(binary_sequence), target_length);
    end
    
    % 返回生成样本的参数信息
    param_info.code_type = code_type;
    param_info.config = config;
    param_info.snr_db = snr_db;
end

%% Hamming码样本生成函数
function binary_sequence = generate_hamming(config, num_frames, target_length, snr_db)
    n = config(1); % 码字长度
    k = config(2); % 信息长度
    
    % 生成随机二进制信息
    info_bits = randi([0, 1], k, num_frames);

    % 根据 n, k 配置选择对应的生成矩阵 G
    if n == 6 && k == 3
        G = [1 0 0 1 1 1;
             0 1 0 1 0 1;
             0 0 1 0 1 1];
    elseif n == 7 && k == 4
        G = [1 0 0 0 1 1 0;
             0 1 0 0 1 0 1;
             0 0 1 0 0 1 1;
             0 0 0 1 1 1 1];
    elseif n == 12 && k == 8
        G = [1 0 0 0 0 0 0 0 1 1 1 1;
             0 1 0 0 0 0 0 0 1 0 1 1;
             0 0 1 0 0 0 0 0 0 1 1 1;
             0 0 0 1 0 0 0 0 1 1 0 1;
             0 0 0 0 1 0 0 0 0 1 1 0;
             0 0 0 0 0 1 0 0 0 0 1 1;
             0 0 0 0 0 0 1 0 0 0 0 1;
             0 0 0 0 0 0 0 1 1 1 1 0];
    elseif n == 15 && k == 11
        G = [1 0 0 0 0 0 0 0 0 1 1 1 1 1 1;
             0 1 0 0 0 0 0 0 0 1 1 1 1 1 0;
             0 0 1 0 0 0 0 0 0 1 1 1 1 0 1;
             0 0 0 1 0 0 0 0 0 1 1 1 0 1 1;
             0 0 0 0 1 0 0 0 0 1 1 1 0 0 1;
             0 0 0 0 0 1 0 0 0 0 1 1 0 1 0;
             0 0 0 0 0 0 1 0 0 0 0 1 1 1 1;
             0 0 0 0 0 0 0 1 0 1 0 1 1 0 0;
             0 0 0 0 0 0 0 0 1 0 1 0 1 1 0;
             0 0 0 0 0 0 0 0 0 1 0 1 0 1 1;
             0 0 0 0 0 0 0 0 0 0 1 0 1 0 1];
    else
        error('Unsupported Hamming code size: [n, k] = [%d, %d]', n, k);
    end
    
    % 将输入的信息比特与生成矩阵相乘得到编码比特
    encoded_bits = mod(info_bits' * G, 2);  % 结果为 num_frames x n
    encoded_bits_flat = encoded_bits(:);    % 展平为一维向量
    
    % 调用BPSK调制解调函数
    binary_sequence = bpsk_mod_demod(encoded_bits_flat, target_length, snr_db);
end

%% BCH码样本生成函数
function binary_sequence = generate_bch(config, num_frames, target_length, snr_db)
    n = config(1); % 码字长度
    k = config(2); % 信息长度
    
    info_bits = randi([0, 1], k, num_frames);

    % 根据 n, k 配置选择对应的生成矩阵 G
    if n == 6 && k == 3
        G= [1 1 0 1 0 0;
            0 1 1 0 1 0;
            0 0 1 1 0 1];
    elseif n == 7 && k == 4
        G = [1 1 0 1 0 0 0;
             0 1 1 0 1 0 0;
             0 0 1 1 0 1 0;
             0 0 0 1 1 0 1];
    elseif n == 12 && k == 8
        G = [1 0 0 1 1 0 0 0 0 0 0 0;
             0 1 0 0 1 1 0 0 0 0 0 0;
             0 0 1 0 0 1 1 0 0 0 0 0;
             0 0 0 1 0 0 1 1 0 0 0 0;
             0 0 0 0 1 0 0 1 1 0 0 0;
             0 0 0 0 0 1 0 0 1 1 0 0;
             0 0 0 0 0 0 1 0 0 1 1 0;
             0 0 0 0 0 0 0 1 0 0 1 1];
    elseif n == 15 && k == 11
        G = [1 0 0 1 1 0 0 0 0 0 0 0 0 0 0;
             0 1 0 0 1 1 0 0 0 0 0 0 0 0 0;
             0 0 1 0 0 1 1 0 0 0 0 0 0 0 0;
             0 0 0 1 0 0 1 1 0 0 0 0 0 0 0;
             0 0 0 0 1 0 0 1 1 0 0 0 0 0 0;
             0 0 0 0 0 1 0 0 1 1 0 0 0 0 0;
             0 0 0 0 0 0 1 0 0 1 1 0 0 0 0;
             0 0 0 0 0 0 0 1 0 0 1 1 0 0 0;
             0 0 0 0 0 0 0 0 1 0 0 1 1 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 1 1 0;
             0 0 0 0 0 0 0 0 0 0 1 0 0 1 1];
    else
        error('Unsupported BCH code size: [n, k] = [%d, %d]', n, k);
    end
    
    encoded_bits = mod(info_bits' * G, 2);
    encoded_bits_flat = encoded_bits(:);
    % 调用BPSK调制解调函数
    binary_sequence = bpsk_mod_demod(encoded_bits_flat, target_length, snr_db);
end

%% Reed-Solomon码样本生成函数
function binary_sequence = generate_rs(config, num_frames, target_length, snr_db)
    n = config(1); % 码字长度
    k = config(2); % 信息长度

    info_bits = randi([0, 1], k, num_frames);

    % 手动构造生成矩阵 G
    if n == 7 && k == 3
        G = [1 0 0 1 1 0 1;
             0 1 0 1 0 1 1;
             0 0 1 0 1 1 1];
    elseif n == 6 && k == 4
        G = [1 0 0 0 1 1;
             0 1 0 0 1 0;
             0 0 1 0 0 1;
             0 0 0 1 1 1];
    elseif n == 7 && k == 5
        G = [1 0 0 0 0 1 1;
             0 1 0 0 0 1 0;
             0 0 1 0 0 0 1;
             0 0 0 1 0 1 1;
             0 0 0 0 1 0 1];
    elseif n == 15 && k == 3
        G = [1 0 0 1 1 0 1 1 0 1 0 1 0 1 1;
             0 1 0 1 0 1 1 0 1 0 1 0 1 1 0;
             0 0 1 0 1 1 0 1 1 1 1 1 0 0 1];
    else
        error('Unsupported RS code size: [n, k] = [%d, %d]', n, k);
    end

    encoded_bits = mod(info_bits' * G, 2);
    encoded_bits_flat = encoded_bits(:);
    % 调用BPSK调制解调函数
    binary_sequence = bpsk_mod_demod(encoded_bits_flat, target_length, snr_db);
end

%% Polar码样本生成函数
function binary_sequence = generate_polar(config, num_frames, target_length, snr_db)
    n = config(1);  % 码字长度
    k = config(2);  % 信息位数

    base_seq = [0 1 2 4 8 3 5 6 9 10 12 16 7 11 13 14 ...
                17 18 20 24 15 19 21 22 25 26 28 31 23 27 29 30];
    reliability_sequence = base_seq + 1;
    info_positions = sort(reliability_sequence(1:k));

    info_bits = randi([0, 1], k, num_frames);

    encoded_bits = zeros(num_frames, n);
    for i = 1:num_frames
        u = zeros(1, n);
        u(info_positions) = info_bits(:, i);

        x = u;
        stages = log2(n);
        for s = 1:stages
            step = 2^s;
            half = step / 2;
            for idx = 1:step:n
                for j = 0:half-1
                    a = idx + j;
                    b = idx + j + half;
                    x(a) = mod(x(a) + x(b), 2);
                end
            end
        end
        encoded_bits(i, :) = x;
    end
    
    encoded_bits_flat = encoded_bits(:);
    % 调用BPSK调制解调函数
    binary_sequence = bpsk_mod_demod(encoded_bits_flat, target_length, snr_db);
end

%% LDPC码样本生成函数
function binary_sequence = generate_ldpc(config, num_frames, target_length, snr_db, matrices)
    n = config(1);
    k = config(2);

    if k ~= 1024
        error('当前函数仅支持信息位长度 k = 1024 的CCSDS LDPC码。');
    end
    
    supported_config = [2048,1024; 1536,1024; 1280,1024];
    if ~any(ismember(supported_config, [n,k], 'rows'))
        error('不支持的[n,k]配置，仅支持[2048,1024]、[1536,1024]、[1280,1024]。');
    end
    
    % 根据码率设置参数
    if n == 2048
        RATE = 1/2;
        M = 512;
    elseif n == 1536
        RATE = 2/3;
        M = 256;
    else
        RATE = 4/5;
        M = 128;
    end

    % 生成校验矩阵 H 和生成矩阵 G
    G = matrices.G;

    % 生成信息比特
    msg_frames = randi([0, 1], k, num_frames);

    % 编码过程
    encoded_frames = zeros(n, num_frames);
    for frame_idx = 1:num_frames
        msg = msg_frames(:, frame_idx);
        encoded_bits = mod(G' * msg, 2);
        if length(encoded_bits) > n
            encoded_bits = encoded_bits(1:n);
        else
            encoded_bits = [encoded_bits; zeros(n - length(encoded_bits), 1)];
        end
        encoded_frames(:, frame_idx) = encoded_bits;
    end

    % 转换为一维向量
    encoded_bits_flat = encoded_frames(:);
    % 调用BPSK调制解调函数
    binary_sequence = bpsk_mod_demod(encoded_bits_flat, target_length, snr_db);
end

function [ H ] = ccsdscheckmatrix2( M ,RATE )
%creating the check matrix of LDPC codes in CCSDS document (version 2,published in 2007)
% input 
%	M: a parameter assign in document CCSDS 131.1-O-2
% 	RATE: information rate

load('checkmatrixconstant.mat');

A = zeros(M);
B = eye(M);

L = 0:M-1;
for matrixNum = 1:26
    t_k = theta(matrixNum);
    f_4i_M = floor(4*L/M);
    f_k = fai{matrixNum}(f_4i_M+1,log2(M)-6)';
    col_1 = M/4*(mod((t_k+f_4i_M),4)) + ...
        mod((f_k+L),M/4);
    row_col = col_1+1 + L*M;
    C_temp = zeros(M);
    C_temp(ind2sub([M,M],row_col)) = 1;
    C{matrixNum} = C_temp';
end

H = [A A B A B+C{1};B B A B C{4}+C{3}+C{2};B C{5}+C{6} A C{7}+C{8} B];

switch(RATE)
    case 1/2
        H=H;
    case 2/3
        H_23 = [A A;C{11}+C{10}+C{9} B; B C{14}+C{13}+C{12}];
        H=[H_23 H];
    case 4/5
        H_23 = [A A;C{11}+C{10}+C{9} B; B C{14}+C{13}+C{12}];
        H_45 = [A A A A;C{23}+C{22}+C{21} B C{17}+C{16}+C{15} B;...
            B C{26}+C{25}+C{24} B C{20}+C{19}+C{18}];
        H = [H_45 H_23 H];
end
end

function [ Gqc ] = ccsdsgeneratematrix2( H,M,RATE )
%creating the generate matrix of LDPC codes in CCSDS document (version 2,published in 2007)
% input 
%   H: check matrix
%	M: a parameter assign in document CCSDS 131.1-O-2
% 	RATE: information rate
switch(RATE)
    case 1/2
        K=2;
    case 2/3
        K=4;
    case 4/5
        K=8;
end

P = H(1:3*M,end-3*M+1:end);
Q = H(1:3*M,1:M*K);
W = mod(inv_bin(P)*Q,2)';
IMK = eye(M*K);
G = [IMK W];

rowNum = 1:M/4:1+M*K-M/4;
Gqc = zeros(size(G));
Gqc(rowNum,:) = G(rowNum,:);
for m = 1:4*K+12
    for n = 1:M/4-1
        Gqc(rowNum+n,M/4*(m-1)+1:M/4*m) = circshift(G(rowNum,M/4*(m-1)+1:M/4*m),n,2);
    end
end
end

function A_inv = inv_bin(A)
    % 二进制矩阵求逆（基于高斯消元法）
    [n, m] = size(A);
    if n ~= m
        error('矩阵必须为方阵才能求逆');
    end
    
    % 构造增广矩阵 [A | I]
    aug = [A, eye(n)];
    
    for i = 1:n
        % 寻找主元
        pivot_row = i;
        while pivot_row <= n && aug(pivot_row, i) == 0
            pivot_row = pivot_row + 1;
        end
        
        if pivot_row > n
            error('矩阵不可逆');
        end
        
        % 交换行
        if pivot_row ~= i
            aug([i, pivot_row], :) = aug([pivot_row, i], :);
        end
        
        % 消去其他行的第i列
        for j = 1:n
            if j ~= i && aug(j, i) == 1
                aug(j, :) = mod(aug(j, :) + aug(i, :), 2);
            end
        end
    end
    
    % 提取逆矩阵
    A_inv = aug(:, n+1:end);
end

%% Turbo码样本生成函数（CCSDS标准）
function binary_sequence = generate_turbo(config,num_blocks, target_length, snr_db)
    n = config(1);  % 码长
    k = config(2);  % 信息位长（1784）
    
    % 计算需要多少个编码块
    num_blocks = ceil(target_length / n);
    
    % 确定模式（根据n和k）
    if n == 3576 && k == 1784
        mode = 1;
    elseif n == 5364 && k == 1784
        mode = 2;
    elseif n == 7152 && k == 1784
        mode = 3;
    elseif n == 10724 && k == 1784
        mode = 4;
    else
        error('不支持的配置');
    end
    
    % 预生成交织表（只需生成一次）
    interleaver_table = gen_interleaver(k);
    
    % 生成随机信息位
    info_bits = randi([0, 1], k, num_blocks);
    
    % 逐块编码
    encoded_bits = zeros(n, num_blocks);
    for i = 1:num_blocks
        % 调用编码函数（传入当前块、模式、交织表）
        full_code = ccsds_turbo_encode(info_bits(:, i)', mode, interleaver_table);
        encoded_bits(:, i) = full_code(1:n);  % 截取码长n
    end
    
    % 展平并处理
    encoded_bits_flat = encoded_bits(:);
    
    % 截取到目标长度
    if length(encoded_bits_flat) > target_length
        encoded_bits_flat = encoded_bits_flat(1:target_length);
    end
    
    binary_sequence =bpsk_mod_demod(encoded_bits_flat, target_length, snr_db);
end

%% 核心编码函数（实现CCSDS编码流程）
function code = ccsds_turbo_encode(info_bits, mode, interleave_table)
    % 步骤1：伪随机化
    scrambled = scramble(info_bits);
    
    % 步骤2：交织
    interleaved = scrambled(interleave_table);
    
    % 步骤3：RSC编码（信息位）
    [p1_info, state1] = rsc_encode_ccsds(scrambled, mode, 1);  % 编码器1
    [p2_info, state2] = rsc_encode_ccsds(interleaved, mode, 2);% 编码器2
    
    % 步骤4：生成终止位
    term1 = get_termination_bits_ccsds(state1, mode, 1);
    term2 = get_termination_bits_ccsds(state2, mode, 2);
    
    % 步骤5：编码终止位
    [p1_term, ~] = rsc_encode_ccsds(term1, mode, 1, state1);
    [p2_term, ~] = rsc_encode_ccsds(term2, mode, 2, state2);
    
    % 步骤6：合并校验位
    p1 = [p1_info p1_term];
    p2 = [p2_info p2_term];
    
    % 步骤7：删余
    parity = puncture_ccsds(p1, p2, mode);
    
    % 步骤8：组合输出（信息位+终止位+校验位）
    code = [scrambled term1 parity];
end

%% 1. 生成CCSDS交织表
function interleaver_table = gen_interleaver(k)
    k1 = 8;
    k2 = k / k1;  % 1784/8=223
    p = [31, 37, 43, 47, 53, 59, 61, 67];
    interleaver_table = zeros(1, k);

    for s = 1:k
        % 计算m
        m = mod(s-1, 2);
        % 计算i
        i = floor((s-1) / (2*k2));
        % 计算j
        j = floor((s-1)/2) - i*k2;
        % 计算t
        t = mod(19*i+1, k1/2);
        % 计算q
        q = mod(t, 8) + 1;
        % 计算c
        c = mod(p(q)*j + 21*m, k2);
        % 计算π(s)
        pi_s = 2*(t + c*k1/2 + 1) - m;
        if pi_s < 1 || pi_s > k
            error(['交织器索引错误：pi_s=', num2str(pi_s), '，超出bits长度=', num2str(k)]);
        end
        interleaver_table(s) = pi_s;
    end
end

%% 2. 伪随机化（扰码）
function scrambled = scramble(data)
    reg = ones(1,8);
    scrambled = zeros(1, length(data));

    for i = 1:length(data)
        % 生成伪随机比特
        prn = reg(8);  % 输出位为x^8（最高位）

        % 计算反馈（多项式：x^8+x^7+x^5+x^3+1）
        feedback = xor(reg(8), reg(7));
        feedback = xor(feedback, reg(5));
        feedback = xor(feedback, reg(3));
        feedback = xor(feedback, 1);  % 常数项1

        % 更新寄存器
        reg = [reg(2:8) feedback];

        % 扰码
        scrambled(i) = xor(data(i), prn);
    end
end

%% 3.RSC编码器（符合CCSDS标准）
function [parity, final_state] = rsc_encode_ccsds(input_bits, mode, enc_id, init_state)
    % CCSDS标准规定的多项式（标准6.3h节）
    % 所有码率共用反向连接向量
    G0 = [1 0 0 1 1];  % 八进制10011
    
    % 根据码率和编码器选择前向连接向量
    if enc_id == 1  % 第一个分量编码器
        switch mode
            case 1  % 1/2码率
                G_forward = {[1 1 0 1 1]};  % 八进制11011
            case 2  % 1/3码率  
                G_forward = {[1 1 0 1 1]};  % 八进制11011
            case 3  % 1/4码率
                G_forward = {[1 0 1 0 1], [1 1 1 1 1]};  % 八进制10101, 11111
            case 4  % 1/6码率
                G_forward = {[1 1 0 1 1], [1 0 1 0 1], [1 1 1 1 1]};  % G1, G2, G3
        end
    else  % 第二个分量编码器
        switch mode
            case 1  % 1/2码率
                G_forward = {[1 1 0 1 1]};  % 八进制11011
            case 2  % 1/3码率
                G_forward = {[1 1 0 1 1]};  % 八进制11011
            case 3  % 1/4码率
                G_forward = {[1 1 0 1 1]};  % 八进制11011
            case 4  % 1/6码率
                G_forward = {[1 1 0 1 1], [1 1 1 1 1]};  % G1, G3
        end
    end
    
    % 初始化状态（4位寄存器）
    if nargin < 4
        init_state = zeros(1, 4);
    end
    state = init_state;
    
    parity = [];
    
    for i = 1:length(input_bits)
        % 计算反馈（使用G0多项式）
        % G0 = [1 0 0 1 1] 对应：输入 + 寄存器3 + 寄存器4
        feedback = mod(input_bits(i) + state(3) + state(4), 2);
        
        % 计算每个前向多项式的校验位
        frame_parity = [];
        for j = 1:length(G_forward)
            G = G_forward{j};
            % G多项式计算：反馈 + 对应寄存器位的加权和
            p_bit = feedback;  % 输入位参与计算
            
            if G(2) == 1  % 寄存器1
                p_bit = mod(p_bit + state(1), 2);
            end
            if G(3) == 1  % 寄存器2  
                p_bit = mod(p_bit + state(2), 2);
            end
            if G(4) == 1  % 寄存器3
                p_bit = mod(p_bit + state(3), 2);
            end
            if G(5) == 1  % 寄存器4
                p_bit = mod(p_bit + state(4), 2);
            end
            
            frame_parity = [frame_parity p_bit];
        end
        
        parity = [parity frame_parity];
        
        % 更新状态：反馈位移入，最旧状态移出
        state = [feedback, state(1:end-1)];
    end
    
    final_state = state;
end

%% 4.终止位生成（符合CCSDS标准）
function term_bits = get_termination_bits_ccsds(state, mode, enc_id)
    % 标准要求：4位终止位，通过反馈路径清零寄存器
    term_bits = zeros(1, 4);
    current_state = state;
    
    for i = 1:4
        % 计算反馈（输入为0时的反馈值）
        feedback = mod(current_state(3) + current_state(4), 2);  % G0 = [1 0 0 1 1]
        term_bits(i) = feedback;
        
        % 更新状态
        current_state = [feedback, current_state(1:end-1)];
    end
end

%% 5.删余操作（符合CCSDS标准）
function parity = puncture_ccsds(p1, p2, mode)
    len = length(p1);
    parity = [];
    
    switch mode
        case 1  % 1/2码率：交替删余
            % 模式：保留所有p1，p2每隔一位删余
            for i = 1:len
                parity = [parity p1(i)];
                if mod(i, 2) == 1  % 奇数位保留p2
                    parity = [parity p2(i)];
                end
            end
            
        case 2  % 1/3码率：无删余
            for i = 1:len
                parity = [parity p1(i) p2(i)];
            end
            
        case 3  % 1/4码率：无删余，但输出顺序不同
            % p1包含2个校验位，p2包含1个校验位
            len1 = length(p1);
            len2 = length(p2);
            
            % 确保有完整的组
            num_groups = min(floor(len1/2), len2);
            
            for i = 1:num_groups
                idx1 = (i-1)*2 + 1;  % p1的第一个校验位索引
                idx2 = i;             % p2的校验位索引
                
                if idx1+1 <= len1 && idx2 <= len2
                    parity = [parity p1(idx1) p1(idx1+1) p2(idx2)];
                end
            end
            
        case 4  % 1/6码率：无删余
            % p1包含3个校验位，p2包含2个校验位
            len1 = length(p1);
            len2 = length(p2);
            
            % 确保有完整的组
            num_groups = min(floor(len1/3), floor(len2/2));
            
            for i = 1:num_groups
                idx1 = (i-1)*3 + 1;  % p1的第一个校验位索引
                idx2 = (i-1)*2 + 1;  % p2的第一个校验位索引
                
                if idx1+2 <= len1 && idx2+1 <= len2
                    parity = [parity p1(idx1) p1(idx1+1) p1(idx1+2) p2(idx2) p2(idx2+1)];
                end
            end
            
    end
end
%% 卷积码样本生成函数
function binary_sequence = generate_convolutional(config, target_length, snr_db)
    constraint_length = config(1);
    output_bits = config(2);
    rate = 1 / output_bits;
    
    % 生成足够的信息比特用于截断
    num_info_bits = ceil(target_length * rate * 1.2);  % 生成稍多的信息比特
    info_bits = randi([0, 1], num_info_bits, 1);

    if constraint_length == 5
        gen = [23 35];
    elseif constraint_length == 6
        gen = [53 75];
    elseif constraint_length == 7
        gen = [133 171];
    elseif constraint_length == 8
        gen = [237 345];
    else
        error('Unsupported constraint length. Use 5 to 8.');
    end

    trellis = poly2trellis(constraint_length, gen);
    encoded_bits = convenc(info_bits, trellis);
    encoded_bits_flat = encoded_bits(:);
    % 调用BPSK调制解调函数
    binary_sequence = bpsk_mod_demod(encoded_bits_flat, target_length, snr_db);
end

%% 通用BPSK调制解调函数（核心修改）
function binary_sequence = bpsk_mod_demod(encoded_bits, target_length, snr_db)
    % 1. BPSK调制：0→-1，1→+1
    modulated_symbols = 2 * encoded_bits - 1;
    
    % 2. 根据SNR计算噪声参数
    Es = 1;  % BPSK符号能量（±1的功率为1）
    snr_linear = 10^(snr_db / 10);  % SNR从dB转线性值
    sigma2 = Es / snr_linear;  % 噪声功率
    sigma = sqrt(sigma2);  % 噪声标准差
    
    % 3. 生成高斯白噪声并叠加到调制符号
    noise = sigma * randn(size(modulated_symbols));  % 服从N(0, sigma²)的噪声
    received_symbols = modulated_symbols + noise;  % 接收信号 = 发送符号 + 噪声
    
    % 4. BPSK硬判决解调：大于0判为1，小于等于0判为0
    demodulated_bits = (received_symbols > 0);
    
    % 5. 截断到目标长度（使用截断方式）
    if length(demodulated_bits) > target_length
        % 截断多余部分
        binary_sequence = demodulated_bits(1:target_length);
    else
        % 如果长度不足，循环重复（但这种情况应该很少发生）
        repeated_seq = repmat(demodulated_bits, ceil(target_length / length(demodulated_bits)), 1);
        binary_sequence = repeated_seq(1:target_length);
    end
    
    % 6. 转换为uint8类型（0/1）
    binary_sequence = uint8(binary_sequence);
end

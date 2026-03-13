import numpy as np

def load_trace(file_path):
    """
    Hàm đọc dữ liệu băng thông từ file text.
    Input: Đường dẫn tới file trace (ví dụ: 'traces/test_trace.txt')
    Output: 2 mảng numpy chứa thời gian và băng thông tương ứng.
    """
    time_all = []
    throughput_all = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parse = line.split() 
            if len(parse) >= 2:
                time_all.append(float(parse[0])) 
                throughput_all.append(float(parse[1])) 
                
    return np.array(time_all), np.array(throughput_all)
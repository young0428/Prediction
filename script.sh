#!/bin/bash

# 루프 횟수 초기화
loop_count=0

# 무한 루프 시작
while true; do
    echo "실행 횟수: $loop_count"
    
    # 파이썬 스크립트 실행 (timeout 1분)
    /bin/python3.11 /host/c/Users/hy105/Desktop/Prediction/Training_specific.py
    
    # 파이썬 스크립트 종료 상태 확인
    exit_status=$?
    
    # 파이썬 스크립트가 timeout에 도달하지 않고 먼저 종료되면 루프 종료
    # if [ $exit_status -ne 124 ]; then
    #     echo "파이썬 스크립트가 먼저 종료되었습니다."
    #     break
    # fi
    
    # 1분 휴식
    pkill -f Training_specific.py
    echo "1분 휴식..."
    sleep 1m
    
    
    # 루프 횟수 증가
    ((loop_count++))
done

echo "루프 종료"

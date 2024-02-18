def merge_intervals_with_gap(intervals, gap_sec):
    merged_intervals = []
    current_interval = None

    for interval in intervals:
        if current_interval is None:
            current_interval = interval[:]
        else:
            # Check if the gap between the current interval and the new one is within the specified limit
            if interval[0] - current_interval[1] <= gap_sec:
                # Merge the intervals
                current_interval[1] = interval[1]
            else:
                # Gap is exceeded, add the current interval and start a new one
                merged_intervals.append(current_interval)
                current_interval = interval[:]

    # Add the last interval
    if current_interval is not None:
        merged_intervals.append(current_interval)

    # Format the output with individual intervals before merging
    final_result = []
    for merged_interval in merged_intervals:
        # Extract individual intervals before merging
        individual_intervals = [interval for interval in intervals if merged_interval[0] <= interval[0] <= merged_interval[1]]
        final_result.append([merged_interval, individual_intervals])

    return final_result

# Example usage:
intervals = [[1, 5], [7, 10], [12, 15], [18, 22], [25, 30]]
gap_sec = 3
result = merge_intervals_with_gap(intervals, gap_sec)
print(result[0])

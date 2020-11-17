import os
import subprocess

def file_path(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)
    return filename

def coords_file(windows_filename, resolution):
    with open(windows_filename) as f:
        assert not type(f) == str
        windows = []

        index = 0
        frame_offset = 0
        for line in f:
            line = line.split()
            chrom, start, end = line[0], int(line[1]), int(line[2])

            if not ((end-start) % resolution) == 0:
                raise Exception("coords must be resolution-aligned!")

            windows.append( [chrom, start, end, index, frame_offset])
            index += 1
            frame_offset += (end-start) // resolution

        num_frames = frame_offset
        return windows, num_frames

def get_frame_index(windows, resolution, chrom, pos):
    for window_index, window in enumerate(windows):
        if ((window[0] == chrom) and
            (window[1] <= pos) and
            (window[2] > pos)):
            window_offset = window[4]
            frame_offset = (pos - window[1]) // resolution
            return window_index, window_offset + frame_offset
    return -1, -1

def runCommand(cmd):
    print(" ".join(cmd))
    subprocess.Popen(cmd).wait()

def maybe_gzip_open(filename, mode="r", *args, **kwargs):
    if filename == "-":
        if mode.startswith("U"):
            raise Exception("U mode not implemented")
        elif mode.startswith("w") or mode.startswith("a"):
            return sys.stdout
        elif mode.startswith("r"):
            if "+" in mode:
                raise Exception("+ mode not implemented")
            else:
                return sys.stdin
        else:
            raise ValueError("mode string must begin with one of"
                             " 'r', 'w', or 'a'")

    if filename.endswith(".gz"):
        return gzip.open(filename, mode, *args, **kwargs)

    return open(filename, mode, *args, **kwargs)

def ceildiv(dividend, divisor):
    "integer ceiling division"
    # int(bool) means 0 -> 0, 1+ -> 1
    return (dividend // divisor) + int(bool(dividend % divisor))
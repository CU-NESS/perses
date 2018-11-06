import subprocess
import numpy as np

def ifreq_from_iframe(iframe, prebuffer, numfreqs):
    """
    """
    if iframe < prebuffer:
        return 0
    elif iframe >= numfreqs + prebuffer:
        return numfreqs - 1
    else:
        return iframe - prebuffer

def pre_slowdown_frames_from_times(frame_rate, slowdown_factor, times):
    """
    """
    float_result =\
        np.ceil(np.array(times) * ((1. * frame_rate) / slowdown_factor))
    return float_result.astype(int)

def slowdown(factor, old_prefix, new_prefix, frame_indices, old_suffix,\
    new_suffix):
    """
    Slows down a set of frames by copying each one (factor-1) times.
    
    factor: the factor by which to slowdown the frames
    """
    if type(factor) not in [int, np.int8, np.int16, np.int32, np.int64]:
        raise ValueError("Factor given to slowdown must be an integer.")
    file_names_same = (old_prefix == new_prefix) and (old_suffix == new_suffix)
    if file_names_same:
        prefix = old_prefix
        suffix = old_suffix
    frame_indices = np.array(frame_indices)
    if factor == 1:
        if file_names_same:
            expected_frame_indices = np.arange(len(frame_indices))
            if np.any(frame_indices != expected_frame_indices):
                temp_prefix = prefix + 'tempTEMPtempTEMPtempTEMP'
                slowdown(1, prefix, temp_prefix, frame_indices, suffix, suffix)
                slowdown(1, temp_prefix, prefix, expected_frame_indices,\
                    suffix, suffix)
        else:
            for iframe in frame_indices:
                old_file_name = old_prefix + str(iframe) + old_suffix
                new_file_name = new_prefix + str(iframe) + new_suffix
                subprocess.call(['mv', old_file_name, new_file_name])
    elif file_names_same:
        temp_prefix = prefix + 'tempTEMPtempTEMPtempTEMP'
        slowdown(factor, prefix, temp_prefix, frame_indices, suffix, suffix)
        new_frame_indices = np.arange(factor * len(frame_indices))
        slowdown(1, temp_prefix, prefix, new_frame_indices, suffix, suffix)
    else:
        for old_iiframe in range(len(frame_indices)):
            old_iframe = frame_indices[old_iiframe]
            old_frame_name = old_prefix + str(old_iframe) + old_suffix
            new_base_frame = (old_iiframe * factor)
            for new_iframe in range(new_base_frame, new_base_frame + factor):
                new_frame_name = new_prefix + str(new_iframe) + new_suffix
                subprocess.call(['cp', old_frame_name, new_frame_name])
            subprocess.call(['rm', old_frame_name])
    return
    

def make_video(video_file_name, frame_rate, frame_prefix, frame_indices,\
    frame_suffix, slowdown_factor):
    """
    """
    slowdown(slowdown_factor, frame_prefix, frame_prefix, frame_indices,\
        frame_suffix, frame_suffix)
    video_command_components =\
    [\
        'ffmpeg',\
        '-r ' + str(frame_rate),\
        '-i ' + frame_prefix + '%d' + frame_suffix,\
        '-b:v 1000k',\
        video_file_name\
    ]
    video_command = ' '.join(video_command_components)
    subprocess.call(video_command.split(' '))


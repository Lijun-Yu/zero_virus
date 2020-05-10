import pandas as pd


def get_output(events):
    output = pd.DataFrame([e._replace(track=None) for e in events])
    columns = ['video_id', 'frame_id', 'movement_id', 'obj_type']
    output = output[columns].sort_values(columns)
    return output

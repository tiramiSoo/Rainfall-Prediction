import pandas as pd

# Tạo DataFrame mẫu

data = {
    'degrees': [10, 45, 80, 120, 160, 190, 250, 300, 360]
}
df = pd.DataFrame(data)

# Định nghĩa hàm để chuyển đổi độ sang hướng
def degrees_to_direction(degrees):
    if degrees >= 0 and degrees < 22.5 or degrees >= 337.5 and degrees <= 360:
        return 'N'
    elif degrees >= 22.5 and degrees < 45:
        return 'NNE'
    elif degrees >= 45 and degrees < 67.5:
        return 'NE'
    elif degrees >= 67.5 and degrees < 90:
        return 'ENE'
    elif degrees >= 90 and degrees < 112.5:
        return 'E'
    elif degrees >= 112.5 and degrees < 135:
        return 'ESE'
    elif degrees >= 135 and degrees < 157.5:
        return 'SE'
    elif degrees >= 157.5 and degrees < 180:
        return 'SSE'
    elif degrees >= 180 and degrees < 202.5:
        return 'S'
    elif degrees >= 202.5 and degrees < 225:
        return 'SSW'
    elif degrees >= 225 and degrees < 247.5:
        return 'SW'
    elif degrees >= 247.5 and degrees < 270:
        return 'WSW'
    elif degrees >= 270 and degrees < 292.5:
        return 'W'
    elif degrees >= 292.5 and degrees < 315:
        return 'WNW'
    elif degrees >= 315 and degrees < 337.5:
        return 'NW'
    elif degrees >= 337.5 and degrees < 360:
        return 'NNW'

# Áp dụng hàm chuyển đổi cho cột 'degrees'
df['direction'] = df['degrees'].apply(degrees_to_direction)

# Hiển thị DataFrame
print(df)

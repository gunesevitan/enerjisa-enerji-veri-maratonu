import sys
from tqdm import tqdm
import pandas as pd
from datetime import timezone
from astral import LocationInfo
from astral.sun import sun

sys.path.append('..')
import settings


if __name__ == '__main__':

    # Using Ankara coordinates since most of the panels are located in there
    city = LocationInfo('Ankara', 'Turkey', 'Europe/Istanbul', 39.57, 32.54)
    df_sun = pd.DataFrame(index=pd.date_range('2019-01-01', '2021-12-31', freq='D'))

    for d in tqdm(df_sun.index):

        # Extract sun features and convert UTC to local timezone
        sun_features = sun(city.observer, date=d)
        sun_features = {k: v.replace(tzinfo=timezone.utc, microsecond=0).astimezone(tz=None).time() for k, v in sun_features.items()}

        for k, v in sun_features.items():
            df_sun.loc[d, k.capitalize()] = v

    df_sun = df_sun.reset_index().rename(columns={'index': 'Date'})
    df_sun.to_csv(settings.DATA / 'sun.csv', index=False)

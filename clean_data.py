"""
@Author: Hanh Cao
"""
import pandas as pd
import numpy as np
import os
import re

BASE_PATH = os.path.join(os.getcwd(), 'IRIData')


class Items(object):
    DRINKS = ['beer', 'carbbev', 'coffee']
    FOODS = ['coldcer', 'fzdinent', 'fxpizza', 'hotdog', 'margbutr', 'mayo', 'milk', 'mustketc',
             'peanbutr', 'saltsnck', 'soup', 'spagsauce', 'sugarsub', 'yogurt']

    def __init__(self, item_info, item_type):
        self.item_info = item_info
        self.item_type = item_type

    def transform_upc(self, x):
        frags = str(x).split('-')
        frags[1] = str(int(frags[1]))
        return "".join(frags)

    def transform_fat_content(self, x):
        """
        The levels of fat, number increases as healthy levels increases:
            1: Full fat
            2: Somewhat reduced fat
            3: Large reduced fat
            4: Fat free
        """
        fat_free = '0 |0G|ZERO|NO|NON|FAT FREE'.split('|')
        unspecified_reduced_fat = 'REDUCED|LESS|LOW|LOWER|LITE|LEAN'.split('|')
        specified_reduced_fat = 'REDUCED|LESS|FAT FREE'.split('|')
        if not isinstance(x, float) and x:
            if (isinstance(x, int) and x == 0) or (any([i in x for i in fat_free]) and '%' not in x):
                return 'fat_free'
            if '%' in x:
                percent_num = float(re.compile('(\d+)%').search(x).group(1))
                if (any([i in x for i in specified_reduced_fat]) and percent_num >= 90) or (
                        not any([i in x for i in specified_reduced_fat]) and percent_num <= 10):
                    return 'large_reduced_fat'
                return 'some_reduced_fat'
            if any([i in x for i in unspecified_reduced_fat]):
                return 'some_reduced_fat'
            return 'full_fat'
        return None

    def transform_calorie_levels(self, x):
        """
        The levels of calories, number increases as the healthy levels increase
            1: Some calories
            2: Less calories
            3: No calories
        """
        calories_free = 'NO|NON|ZERO|DIET'.split('|')
        less_calories = 'FEWER|LOW|LIGHT|LITE'.split('|')
        if not isinstance(x, float) and x:
            if (isinstance(x, int) and x == 0) or (any([i in x for i in calories_free])):
                return 'no_calories'
            if any([i in x for i in less_calories]):
                return 'less_calories'
            return 'some_calories'
        return None

    def transform_sugar_content(self, x):
        """
        The levels of sugar content:
            'sugar': Sugar
            'less sugar': Less sugar
            'organic sugar': Organic sugar
            'no sugar': No sugar
        """
        sugar_free = 'NO|NON|ZERO|UNSWEETENED|FREE'.split('|')
        less_sugar = 'LESS|PERCENT'.split('|')
        if isinstance(x, str):
            if any([i in x for i in sugar_free]):
                return 'no_sugar'
            elif 'ORGANIC CANE' in x:
                return 'organic_sugar'
            elif any([i in x for i in less_sugar]):
                return 'less_sugar'
            return 'sugar' if x else None
        return

    def transform_caffeine_info(self, x):
        """
        The levels of caffeine:
            'caffeine': Caffeine
            'less_caffeine': Less caffeine
            'decaffeinated': No caffeine
        """
        caffeine_free = 'ZERO|NO|DECAFFEINATED|FREE'.split('|')
        if not isinstance(x, str):
            return None
        if any([i in x for i in caffeine_free]):
            if '%' not in x:
                return 'decaffeinated'
            return 'less_caffeine'
        return 'caffeine'

    def transform_user_info(self, x):
        """
        Products user are categorized into:
            'adult_both', 'adult_female', 'adult_male', 'baby', 'family', 'youth'
        """
        if not isinstance(x, str):
            return None
        if 'FAMILY' in x or ('ADULT' in x and ('&' in x or 'and' in x)) or 'ASSORTED' in x:
            return 'family'
        elif 'INFANT' in x or 'BABY' in x:
            return 'baby'
        elif any([i in x for i in ['YOUTH', 'KIDS', 'CHILDREN', 'CHILD']]):
            return 'youth'
        elif 'MEN AND WOMEN' in x:
            return 'adult_both'
        elif any([i in x for i in ['MAN', 'MENS', 'MEN']]):
            return 'adult_male'
        elif any([i in x for i in ['WOMEN', 'WOMENS', 'WOMAN']]):
            return 'adult_female'
        return None

    def clean(self):
        items_columns = list(self.item_info.columns)

        # Replace missing, invalid values to None
        self.item_info = self.item_info.replace('MISSING', None)

        # Convert UPC to COLUPC for latter matching with panels_items
        self.item_info["COLUPC"] = self.item_info["UPC"].apply(lambda x: self.transform_upc(x))
        self.item_info["COLUPC"] = self.item_info["COLUPC"].astype(np.int64)

        # Add "PRODUCT SUBTYPE", which is subcategory of "PRODUCT TYPE"
        type_column = [col for col in items_columns if "TYPE OF" in col and col != 'TYPE OF SWEETENER']
        self.item_info["product_subtype"] = self.item_info[type_column[0]] if len(type_column) > 0 else None

        # Rename the first column 'L1' to 'CATEGORY'
        self.item_info["category"] = self.item_info["L1"].apply(lambda x: x.split('-')[1].strip())

        # Transform fat content to 4 different fat levels
        if self.item_type in self.FOODS and 'FAT CONTENT' in items_columns:
            self.item_info['fat_level'] = self.item_info['FAT CONTENT'].apply(lambda x: self.transform_fat_content(x))

        # Transform calorie level to 3 different levels
        if 'CALORIE LEVEL' in items_columns:
            self.item_info['calorie_level'] = self.item_info['CALORIE LEVEL'].apply(
                lambda x: self.transform_calorie_levels(x))

        # Transform sugar content to 3 different levels
        if 'SUGAR CONTENT' in items_columns:
            self.item_info['sugar_level'] = self.item_info['SUGAR CONTENT'].apply(
                lambda x: self.transform_sugar_content(x))
        if 'TYPE OF SWEETENER' in items_columns:
            self.item_info['sugar_level'] = self.item_info['TYPE OF SWEETENER'].apply(
                lambda x: self.transform_sugar_content(x))

        # Transform caffeine info to 2 different levels
        if 'CAFFEINE INFO' in items_columns:
            self.item_info['caffeine_level'] = self.item_info['CAFFEINE INFO'].apply(
                lambda x: self.transform_caffeine_info(x))

        # Transform user info to more concrete categories
        if 'USER INFO' in items_columns:
            self.item_info['user_info'] = self.item_info['USER INFO'].apply(lambda x: self.transform_user_info(x))

        # Only keep those necessary columns
        self.item_info = self.item_info.loc[:, self.item_info.columns.isin(["COLUPC", "category", "PRODUCT TYPE",
                                                                            "fat_level", "calorie_level", "sugar_level",
                                                                            "caffeine_level", "product_subtype",
                                                                            "user_info", "FLAVOR/SCENT", "VOL_EQ"])]

        # There shouldn't be duplicates, but drop if there is
        self.item_info = self.item_info.drop_duplicates()
        return self.item_info


def extract_panels_items():
    # Extract panelists and bought items data
    panels_items = None
    for root, dirs, files in os.walk(os.path.join(BASE_PATH, 'Year12'), topdown=False):
        df = None
        week = None
        for file_name in files:
            if file_name.endswith(".DAT"):
                df = pd.read_csv(os.path.join(root, file_name), sep=',', index_col=False, engine='python')
                df = df.groupby(['PANID', 'COLUPC', 'WEEK'])['UNITS'].sum().reset_index(name='total_purchase')
            if 'IRI week translation' in file_name:
                week = pd.read_excel(os.path.join(root, file_name), sheet_name="Sheet1")
                week['month'] = week['Calendar week starting on'].apply(lambda x: x.month_name())
                week['week'] = week['Calendar week starting on'].apply(lambda x: int(x.weekofyear))
                week = week[['IRI Week', 'month', 'week']]
        if isinstance(df, pd.DataFrame) and isinstance(week, pd.DataFrame):
            df = pd.merge(df, week, how="left", left_on="WEEK", right_on="IRI Week")
            df.drop(['IRI Week'], axis=1, inplace=True)
            if panels_items is None:
                panels_items = df
            else:
                panels_items = pd.concat([panels_items, df], ignore_index=True)
    return panels_items


def extract_panels_demo():
    # Extract panelists demographic info
    panels_demos = pd.read_csv(os.path.join(BASE_PATH, "ads demos12.csv"))

    # Drop these columns as those are supposed to be removed or not helpful
    panels_demos.drop(['HH_LANG', 'MICROWAVE', 'device_type', 'ALL_TVS', 'CABL_TVS', 'Hispanic Flag',
                       'HISP_CAT', 'RACE2', 'RACE3', 'Household Head Race', 'market based upon zipcode',
                       'EXT_FACT', 'IRI Geography Number'],
                      axis=1, inplace=True)
    return panels_demos


def extract_items():
    # Extract items info
    items = None
    for root, dirs, files in os.walk(os.path.join(BASE_PATH, 'parsed stub files 2012'), topdown=False):
        for file_name in files:
            df_info = file_name.split('.')[0].split('_')[1]
            print(f"** {df_info}")
            df = pd.read_excel(os.path.join(root, file_name), sheet_name="Sheet1")
            df = Items(df, df_info).clean()
            if items is None:
                items = df
            else:
                items = pd.concat([items, df], ignore_index=True)
    return items


def extract_data():
    cleaned_path = os.path.join(BASE_PATH, 'cleaned_data')
    if not os.path.isdir(os.path.join(cleaned_path)):
        os.makedirs(cleaned_path)

    print("Extracting panelists demography...")
    panels_demos = extract_panels_demo()
    panels_demos.to_csv(f"{cleaned_path}/panels_demos.csv", index=False)

    print("Extracting panelists and their past purchases...")
    panels_items = extract_panels_items()
    panels_items.to_csv(f"{cleaned_path}/panels_items.csv", index=False)

    print("Extracting items info...")
    items = extract_items()
    items.to_csv(f"{cleaned_path}/items.csv", index=False)

    panels_items_full = pd.merge(panels_items, items, left_on='COLUPC', right_on='COLUPC', how='inner')
    panels_items_full.to_csv(f"{cleaned_path}/panels_items_full.csv", index=False)


if __name__ == "__main__":
    extract_data()

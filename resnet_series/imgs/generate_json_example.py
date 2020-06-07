"""
 * @Author: yufeng.Chen 
 * @Date: 2020-06-07 15:42:03 
 * @Last Modified by: yufeng.Chen
 * @Last Modified time: 2020-06-07 15:49:58
 """
import os
import json
import glob


class JsonOperation:
    @staticmethod
    def write(json_path, data, *args, **kwargv):
        """Save data to json file.

        Args:
            json_path (str): ur path to json loc.
            data (dict: data in dict() format

            example:
                data = dict()
                data['imgA'] = dict()
                data['imgA']['isdog'] = True
                data['imgA']['iscat'] = False
        """

        with open(json_path, 'w') as f:
            json.dump(data, f)
        f.close()
        return

    @staticmethod
    def read(json_path, *args, **kwargv):
        """Read json file.

        Args:
            json_path (str): path to json loc.

        Raises:
            ValueError: raise if file not exist

        Returns:
            dict: json content in dictionary format.
        """
        if not os.path.exists(json_path):
            raise ValueError(
                '{:} path does not exists'.format(json_path))
        with open(json_path) as f:
            data = json.load(f)
        f.close()
        return data


if __name__ == '__main__':
    imgPaths = './imgs/*.jpg'
    myjson = './imgs/label.json'
    
    data = dict()

    for path in glob.glob(imgPaths):
        _, name = os.path.split(path)
        data[name] = dict()

        data[name]['dog'] = 'dog' in path
        data[name]['cat'] = 'cat' in path
    JsonOperation.write(myjson, data)
    
        

import os
import logging

class loggerOperation:
    @staticmethod
    def init(path=None, *args, **kwargv):
        if path is None:
            node_, name_ = os.path.split(__file__)
            path = os.path.join(node_, os.path.splitext(name_)[0]) + '.log'
        # set up logging to file - see previous section for more details
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=path,
                            filemode='a+')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-6s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        logging.propagate = True


        return logging
        

if __name__ == '__main__':
    print(__file__)
    a = CuzLog('./logs/model.log')
    text = 'sssssssssssssssssssssssssssssssadasdaaaaaaaaaaaaaaasdasdsadasas'
    a.info(text)
    a.info(' ' + text)
    a.warning('  ' + text)

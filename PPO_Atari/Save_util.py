
class Save_file_util(object):
    arg_names = []

    @classmethod
    def push(self, x):
        Save_file_util.arg_names.append(x)

    @classmethod
    def get_file_name(self, head, tail=""):
        path = head + "_"

        for par in Save_file_util.arg_names:
            path += "_" + str(par)

        path = path + tail

        return path

    @classmethod
    def create_result(self, result, head=None, tail=""):
        if head is None:
            path = ""
        else:
            path = head + "_"

        for par in Save_file_util.arg_names:
            path += "_" + str(par)

        for par in result:
            path += "_" + str(par)



        with open(path,"w"):pass




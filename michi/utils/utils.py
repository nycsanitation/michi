def drop_consecutive_duplicates(a):
    return [
        a[i] for i in range(len(a))
        if (i == (len(a) - 1)) or (a[i] != a[i+1])
    ]

def method_file_cache(filename):
    """
    A decorator that caches the results of object methods on disk as a pickle
    file. The object must have the following attributes:

        self.cach_path - a Pathlib.Path that points to the directory to use
        self.force_rebuild - boolean, if True, don't use the cache

    Parameters
    ----------
    filename : str
        The name of the pickle file
    """
    def decorator(function):
        def wrapper(*args, **kwargs):
            self = args[0]
            cache_path = self.cache_path / filename

            if cache_path.exists() and not self.force_rebuild:
                with cache_path.open('rb') as pickle_file:
                    try:
                        return pickle.load(pickle_file)
                    except:
                        pass

            result = function(*args, **kwargs)
            with cache_path.open('wb') as pickle_file:
                pickle.dump(result, pickle_file)

                return result

        return wrapper

    return decorator

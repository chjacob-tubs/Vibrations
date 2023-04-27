import Vibrations as vib


def test_fancy_box():
    # Arrange
    string = "TEST"
    # Act
    try:
        vib.fancy_box(string)
        test = True
    except:
        test = False
    # Assert
    assert test == True   



def test_do_cprofile():
    # do_cprofile is a decorator
    # Arrange
    a = 1
    # Pre-Act
    @vib.do_cprofile
    def _test_func(a):
        b = 2
        print('b=',b)
        print('a=',a)
        return a, b
    # Act
    try:
        _test_func(a)
        test = True
    except:
        test = False
    # Assert
    assert test == True



def test_timefunc():
    # do_cprofile is a decorator
    # Arrange
    a = 1
    # Pre-Act
    @vib.timefunc
    def _test_func(a):
        b = 2
        print('b=',b)
        print('a=',a)
        return a, b
    # Act
    try:
        _test_func(a)
        test = True
    except:
        test = False
    # Assert
    assert test == True

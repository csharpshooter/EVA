class TrainHelper:

    def cyclical_lr(maxlrepoch, epochs, minlr, maxlr):

        # Additional function to see where on the cycle we are
        def calculatelrforepoch(it, maxlrepoch, epochs, minlr, maxlr):
            delta = maxlr - minlr
            deltaone = delta / (maxlrepoch - 1)
            deltatwo = delta / (epochs - maxlrepoch)

            if it < maxlrepoch:
                val = minlr + (deltaone * it)
                return val
            else:
                val = maxlr - ((it - maxlrepoch + 1) * deltatwo)
                return val

        lr_lambda = lambda it: calculatelrforepoch(it, maxlrepoch, epochs, minlr, maxlr)
        return lr_lambda

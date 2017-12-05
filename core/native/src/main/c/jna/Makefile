GXX := gcc
FILES = test binToBestSplit
dir = ../../../../../../src/main/resources/linux-x86-64

all:
	for i in `echo $(FILES)`; \
	do \
		echo $$i; \
		$(GXX) -fPIC -c "$$i".c; \
		$(GXX) -shared -o $(dir)/lib"$$i".so "$$i".o; \
	done

clean:
	rm *.o *.dylib *.a *.dll *.so

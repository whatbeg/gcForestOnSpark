GXX := gcc
FILES = core_JNITest jnibinToBestSplit
dir = ../../../../../../src/main/resources/linux-x86-64

all:
	for i in `echo $(FILES)`; \
	do \
		echo $$i; \
		$(GXX) -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux -fPIC -c "$$i".c; \
		$(GXX) -I${JAVA_HOME}/include -I${JAVA_HOME}/include/linux -shared -o $(dir)/lib"$$i".so "$$i".o; \
	done

clean:
	rm *.o *.dylib *.a *.dll *.so

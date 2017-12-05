/*
 * Copyright 2017 Authors NJU PASA BigData Laboratory. Qiu Hu. huqiu00#163.com
 */
package org.apache.spark.ml.core;

import com.sun.jna.*;

import java.util.ArrayList;
import java.util.List;

public class JNA {
    public interface Lgetlib extends Library {
        Lgetlib INSTANCE = (Lgetlib) Native.loadLibrary("test", Lgetlib.class);
        int add(int a, int b);
        void sayUser(UserStruct.ByReference struct);
    }
    public int add(int a, int b) {
        return Lgetlib.INSTANCE.add(a, b);
    }
    public void sayUser(UserStruct.ByReference struct) { Lgetlib.INSTANCE.sayUser(struct); }

    public static class UserStruct extends Structure {
        public NativeLong id;
        public WString name;
        public int age;

        @Override
        protected List<String> getFieldOrder() {
            List<String> a = new ArrayList<String>();
            a.add("id");
            a.add("name");
            a.add("age");
            return a;
        }

        public static class ByReference extends UserStruct implements Structure.ByReference { }
        public static class ByValue extends UserStruct implements Structure.ByValue { }
    }

    public static void main(String[] args) {
        JNA ts = new JNA();
        int c = ts.add(10, 20);
        System.out.println("10 + 20 = " + c);

        UserStruct.ByReference userStruct = new UserStruct.ByReference();
        userStruct.id = new NativeLong(100);
        userStruct.age = 30;
        userStruct.name = new WString("aobama");
        ts.sayUser(userStruct);
    }
}

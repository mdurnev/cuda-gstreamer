#!/usr/bin/env python2

import os
import sys
import re

if __name__ == "__main__":

    #----------------
    # run autogen.sh
    #----------------
    if os.system("./autogen.sh") != 0:
        print("Error: autogen.sh failed")
        sys.exit(2)

    #------------------
    # parse config.log
    #------------------
    try:
        f = open("config.log", "r")
    except IOError as err:
        print(str(err))
        sys.exit(2)

    q_cflags = re.compile(r"^GST_CFLAGS=[\"\'](.*)[\"\']")
    q_libs   = re.compile(r"^GST_LIBS=[\"\'](.*)[\"\']")

    cflags = []
    lflags = []

    for l in f.readlines():
        m = q_cflags.match(l)
        if m:
            cflags = m.group(1).split()
        else:
            m = q_libs.match(l)
            if m:
                lflags = m.group(1).split()
        if cflags != [] and lflags != []:
            break

    f.close()

    include_dirs = []
    libs = []

    for flag in cflags:
        if len(flag) > 2 and flag[0:2] == "-I":
            include_dirs.append(flag[2:])

    if len(include_dirs) <= 0:
        print("Warning: No include dirs")

    for flag in lflags:
        if len(flag) > 2 and flag[0:2] == "-l":
            libs.append(flag[2:])
    
    if len(libs) <= 0:
        print("Warning: No libs")

    del cflags
    del lflags

    #----------------------
    # Create project files
    #----------------------
    q_tbd_includes = re.compile(r"^(.*)TBD_INCLUDES(.*)$")
    q_tbd_libs     = re.compile(r"^(.*)TBD_LIBS(.*)$")

    dir_list = os.listdir(".")
    for i in dir_list:
        if i[-3:] == ".cu": 
            f_in = open("%s/.cproject.in" % i, "r")
            if not f_in:
                print("Warning: %s/.cproject.in does not exist" %i)
                break

            f_out = open("%s/.cproject" % i, "w")

            for j in f_in.readlines():
                m = q_tbd_includes.match(j)
                if m:
                    prefix = m.group(1)
                    postfix = m.group(2)
                    for k in include_dirs:
                        f_out.write("%s%s%s\n" % (prefix, k, postfix))
                else:
                    m = q_tbd_libs.match(j)
                    if m:
                        prefix = m.group(1)
                        postfix = m.group(2)
                        for k in libs:
                            f_out.write("%s%s%s\n" % (prefix, k, postfix))
                    else:
                        f_out.write(j)

            f_out.close()
            f_in.close()


def convert(file_name):
    """
    converts file (file_name is a string, ending in txt or tex)
    to a desired version of it
    """

    file_name_no_extension = file_name[:-4]

    with open(file_name,"r") as f:
        contents = f.read()


    new_contents = fbox_to_kb(contents)


    new_file_name = file_name_no_extension + "_FINAL.tex"

    with open(new_file_name,"w") as n_f:
        n_f.write(new_contents)


def fbox_to_kb(contents):
    """
    contents is a string
    """

    contents_trimmed = contents[41:]

    new_contents = contents

    titles = []
    texts = []


    for i in range(len(contents_trimmed)):

        if contents[i:i+41] == "\\fbox{\\begin{minipage}[t]{1\\columnwidth}%":
            title = ""
            if contents[i+42:i+49] == "\\uline{":
                for j in range(i+49,len(contents)):
                    if contents[j] == "}":
                        break
                    else:
                        title += contents[j]
                # print(title)
            titles.append(title)

            text = ""
            for k in range(i+49+len(title)+1,len(contents)-15):
                if contents[k:k+15] == "\\end{minipage}}":
                    break
                text += contents[k]
            texts.append(text)
            # print(text)


    print("TITLES:",titles)
    print("TEXTS:",texts)

    title_no = -1


    for i in range(len(contents_trimmed)):


        if contents[i:i+41] == "\\fbox{\\begin{minipage}[t]{1\\columnwidth}%":


            differential = len(new_contents) - len(contents)

            print("DIFFERENTIAL","#"+str(title_no+1)+":",differential)

            new_contents = new_contents[:i+differential] + "\\begin{kb}{\\kww{" + new_contents[i+differential+41:]

            if contents[i+42:i+49] == "\\uline{":
                title_no +=1

                new_contents = new_contents[:i+differential+16] + titles[title_no] + "}} a" + new_contents[i+differential+15+len(titles[title_no])+8+2:]




            beg_endix = i+49+len(titles[title_no])+1+len(texts[title_no])
            end_endix = beg_endix + 15



            if contents[beg_endix:end_endix] == "\\end{minipage}}":



                new_beg_endix = i+differential+16+len(titles[title_no])+4+len(texts[title_no])
                new_end_endix = new_beg_endix + 15 + 1

                new_contents = new_contents[:new_beg_endix] + "\\end{kb}" + new_contents[new_end_endix:] 


    return new_contents










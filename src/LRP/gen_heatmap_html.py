import pickle


def gen_html(heatmap):
    prefix = "<html><head></head>\n<body>"

    def value2color(value):
        cap = 1
        if value > 0:
            if value > cap:
                value = cap
            other_color = 255 - int(255 * (value / cap))
            color = "#ff{:02x}{:02x}".format(other_color, other_color)
        else:
            if value < -cap:
                value = -cap
            other_color = 255 - int(255 * (value / -cap))
            color = "#{:02x}{:02x}ff".format(other_color, other_color)
        return color
    unk_count = 0
    content = ""
    for (word, value) in heatmap:
        if word == "<UNK>":
            unk_count= unk_count+ 1
            if unk_count > 3:
                break
        else:
            unk_count = 0
        color = value2color(value)
        content += "<span style=\"background-color: {}\" value={}>{}</span>&nbsp\n".format(color, value, word)

    postfix = "\n</body></html>"
    return prefix + content + postfix


def draw_heatmap():
    heatmaps = pickle.load(open("heatmap.pickle", "rb"))
    path = "C:\work\Data\heatmap\\"
    for idx in [1,2,3,4,5]:
        topic, heat = heatmaps[idx]
        if topic is None:
            continue
        print(topic)
        filename = topic + "{}.html".format(idx)
        open(path + filename, "w").write(gen_html(heat))


draw_heatmap()
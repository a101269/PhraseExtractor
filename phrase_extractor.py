# encoding:utf-8
#    Author:  a101269
#    Date  :  2022/10/14
import hanlp
import re

"""环境：需要hanlp 2.1以上版本"""

class PhraseExtractor():
    def __init__(self, posflag=True, depflag=False, nerflag=False):
        self.tokenizer = hanlp.load(
            hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
        self.tasks = ["tok/coarse"]
        if posflag:
            self.tasks.append("pos/ctb")
        if nerflag:
            self.tasks.append("ner/pku")
        if depflag:
            self.tasks.append("dep")
        self.s_patt = re.compile('\s')
        self.dig_char_patt = re.compile('[a-zA-Z\d]')

    def parse_offset(self, text, tokens):
        if not bool(text) or not bool(tokens):
            return []

        # tokens 总是比text内容少，因为去掉了空格等无用字符
        offsets = []
        off = 0  # text的当前偏移量
        k = 0
        while off < len(text) and k < len(tokens):
            s = tokens[k]
            # 首字母需要相同
            while off < len(text) and text[off] != s[0]:
                assert (text[off].isspace())
                off += 1
                continue
            # 后面的token 需要 相同
            assert (text[off:off + len(s)] == s)
            offsets.append((off, off + len(s)))  # 实际的偏移量
            k += 1
            off += len(s)
        return offsets

    def tokenize(self, texts):
        res = self.tokenizer(texts, tasks=self.tasks)
        batch_tokens = res["tok/fine"]
        res["tok/offsets"] = [self.parse_offset(text, tokens) for text, tokens in zip(texts, batch_tokens)]
        return res

    def preprocess(self, sent):
        ss = re.finditer(self.s_patt, sent)
        sub_flags = []
        for obj in ss:
            if obj.start() > 0 and re.match(self.dig_char_patt, sent[obj.start() - 1]):
                sub_flags.append(-1)
            else:
                sub_flags.append(obj.start())
        sent = list(sent)
        for posi in sub_flags:
            if posi > -1:
                sent[posi] = ""
        return ''.join(sent)

    def rule_based_phrase_extract(self,texts,min_len=3):
        texts=[self.preprocess(sent) for sent in texts]
        res = []
        try:
            han=self.tokenizer(texts, tasks=["tok/coarse", "pos", "dep","sdp"], skip_tasks='tok/fine')
            for si, sent in enumerate(texts):
                words = han["tok/coarse"][si]
                pos = han['pos/ctb'][si]
                dep = han["dep"][si]
                sdp = han["sdp"][si]
                r_dep = {}
                for i, head in enumerate(dep):
                    if head[0] not in r_dep:
                        r_dep[head[0]] = []
                    r_dep[head[0]].append([i + 1, head[1]])

                nn_spans = self.get_nn_span(pos, words, dep, sdp)
                phrases = self.get_phrase(words, nn_spans, pos, dep, r_dep, sdp,min_len=min_len)
                res.append(phrases)
        except:
            return res
        return res

    def get_nn_span(self, pos, words, dep, sdp):
        spans = []
        cur_pot = -1
        for i, p in enumerate(pos):
            word = words[i]
            if i <= cur_pot:
                continue
            if i == 0:
                tmp = []
            DescF = False
            for sd in sdp[i]:
                if sd[1] in ["dDesc"]:#,"Desc"]:
                    DescF = True
                    break
            if dep[i][1] == 'nn' or DescF:
                if p == "NT" and re.search("\d[月日时分]", words[i]):
                    cur_pot = dep[i][0] - 1
                else:
                    brF = False
                    for j in range(i, dep[i][0]):
                        tmp.append(j)
                        if (pos[j] in ["DEG", "DEC", "ETC"] and words[j] != "之") or (
                                pos[j] == "NT" and re.search("\d[月日时分]", words[j])) or (
                                j < dep[i][0] - 1 and dep[j][0] - 1 < tmp[0]) or words[j] == "、":
                            tt = []
                            for t in tmp:
                                if (pos[t][0] == "N" or pos[t] == "JJ") and not (pos[t] == "NT" and re.search("\d[月日时分]", words[t])):
                                    tt.append(t)
                                    if t < len(pos) - 3 and words[t + 1] == "（" and re.match("[a-zA-Z]", words[t + 2]):
                                        tt += [t + 1, t + 2, t + 3]
                                else:
                                    break
                            tmp = tt
                            brF = True
                            break
                    isTm = False
                    if not brF and bool(tmp) and sdp[tmp[-1]][0][1] in ["Time", "Tdur"]:  # 1
                        tmp = tmp[:-1]
                        isTm = True
                    cur_pot = tmp[-1] if bool(tmp) and tmp[-1] < dep[i][0] - 1 else dep[i][0] - 1  # 2
                    if isTm:
                        cur_pot += 2
                    if not bool(tmp):
                        cur_pot = i
            elif p[0] == 'N':
                if p == "NT" and re.search("\d[月日时分]", words[i]):
                    pass
                else:
                    tmp.append(i)
                    if dep[i][1] == "conj" and dep[i][0] == i + 3:
                        tmp += [i + 1, i + 2]
                        cur_pot = i + 2
                    elif i < len(pos) - 3 and words[i + 1] == "（" and re.match("[a-zA-Z]", words[i + 2]):
                        tmp += [i + 1, i + 2, i + 3]
                        cur_pot = i + 3
            elif p == 'JJ':
                tmp.append(i)
            elif p == "OD":
                if i + 2 < len(pos):
                    if pos[i + 1][0] == "M" and pos[i + 2][0] == "N":
                        tmp += [i, i + 1, i + 2]
                        cur_pot = i + 2
                elif i + 1 < len(pos) and pos[i + 1][0] == "N":
                    tmp += [i, i + 1]
            elif p == "CD":
                if i + 1 < len(pos) and (pos[i + 1][0] == "N" or pos[i + 1] == "JJ") and re.match(
                        "[\d一单双两二三四五六七八九十百千万兆亿多少]", words[i]) and words[i] != "一些":
                    tmp += [i, i + 1]
            elif p == "AD" and i < len(pos) - 1 and pos[i + 1][0] == "V" and dep[i + 1][1] in ["dep", "advmod", "vmod",
                                                                                               "root"] and bool(tmp) and \
                    dep[i + 1][0] - 1 >= tmp[0]:
                if bool(tmp):
                    tmp.append(i)
            elif p[0] == "V" and i > 0 and pos[i - 1] == "AD" and dep[i][1] in ["dep", "advmod", "vmod",
                                                                                "root"] and bool(tmp) and dep[i][
                0] - 1 >= tmp[0]:
                if bool(tmp):
                    tmp.append(i)
            else:
                if bool(tmp):
                    if pos[tmp[0]] == 'AD':
                        tmp = []
                    else:
                        spans.append(tmp)
                        tmp = []
            if (i == len(pos) - 1 or cur_pot == len(pos) - 1) and bool(tmp):
                if pos[tmp[0]] == 'AD':
                    pass
                else:
                    spans.append(tmp)
        return spans

    def get_phrase(self, words, nn_spans, pos, dep, r_dep, sdp,min_len=None):
        phrases = {}
        blanks = []
        # print(nn_spans)
        for span in nn_spans:
            span = list(set(span))
            span.sort()
            tmp = set()
            cur_pot = 0
            for id in span:
                word = words[id]
                if id < cur_pot:
                    continue
                if len(span) > 1 and dep[id][0] - 1 in span and dep[id][1] == "nn":
                    for j in list(range(id, dep[id][0])):
                        tmp.add(j)
                    cur_pot = dep[id][0] - 1
                elif dep[id][0] - 1 in span and dep[id][1] in ["nsubj", "dobj", "amod", "vmod", "tmod", "nummod",
                                                               "advmod", "rcmod", "ordmod", "seq", "clf", "conj", "cc",
                                                               "punct", "prnmod", "assm"]:
                    tmp.add(id)
                    tmp.add(dep[id][0] - 1)
                elif len(span) > 1 and dep[id][1] == "nn" and dep[id][0] - 1 > span[-1]:
                    for j in list(range(id, span[-1] + 1)):
                        tmp.add(j)
                    cur_pot = dep[id][0] - 1
                if dep[id][1] in ["nn", "dobj", "amod"] and dep[id][0] - 1 - span[-1] == 1 and pos[dep[id][0] - 1][
                    0] != "V":
                    tmp.add(id)
                    tmp.add(dep[id][0] - 1)
                elif dep[id][1] in ["nsubj"] and dep[id][0] - 1 - span[-1] == 1 and pos[dep[id][0] - 1] == "VV":
                    add_f = False
                    for tail in r_dep[dep[id][0]]:
                        if tail[1] in ['dobj', 'ccomp', 'rcomp', 'range', 'pobj', 'lobj', 'comod', 'pccomp']:
                            add_f = False
                            break
                        elif dep[dep[id][0] - 1][1] in ['pccomp','ccomp','rcomp']:
                            add_f = False
                            break
                        else:
                            for sd in sdp[id]:
                                if sd[1] in ["Exp", "dExp"]:
                                    add_f = True
                                    break
                            if not add_f:
                                for sd in sdp[dep[id][0] - 1]:
                                    if sd[1] in ["dDesc", "dCont"]:
                                        add_f = True
                                        break
                    if add_f:
                        tmp.add(id)
                        tmp.add(dep[id][0] - 1)
            if not bool(tmp) and len(span) == 1:
                if len(words[span[0]]) > 3 and pos[span[0]] in ["NN", "NR"]:
                    tmp.add(span[0])

            tmp = list(tmp)
            tmp.sort()
            if bool(tmp) and tmp[0] > 0 and words[tmp[0] - 1][0] == "全" and pos[tmp[0] - 1] == 'DT':
                tmp = [tmp[0] - 1] + tmp
            ph = ''
            ph_ids = []
            blanks.append(0)
            if len(tmp) > 1:
                if pos[tmp[0]] == 'AD':
                    continue
                elif pos[tmp[-2]] == "AD":
                    tmp = tmp[:-2] if len(tmp) > 3 else []

            for i, id in enumerate(tmp):
                if i > 0 and id - tmp[i - 1] > 1:
                    if bool(ph):
                        if ph not in phrases:
                            phrases[ph] = {"word_offset": [], "char_offset": []}
                        phrases[ph]["word_offset"].append([ph_ids[0], ph_ids[-1]])
                        w_off_s = len(''.join([w for w in words[:ph_ids[0]]]))
                        w_off_e = len(''.join([w for w in words[ph_ids[0]:ph_ids[-1] + 1]]))
                        phrases[ph]["char_offset"].append([w_off_s, w_off_e])
                        ph = ""
                        ph_ids = []
                if bool(ph):
                    if re.match("[a-zA-Z]", ph[-1]) and re.match("[a-zA-Z]", words[id]):
                        ph += " "
                        blanks[-1] += 1
                ph += words[id]
                ph_ids.append(id)
            if bool(ph):
                if len(ph) < 3:
                    continue
                if re.search("”", ph) and not re.search("“", ph):
                    ph = "“" + ph
                if ph not in phrases:
                    phrases[ph] = {"word_offset": [], "char_offset": []}
                phrases[ph]["word_offset"].append([ph_ids[0], ph_ids[-1]])
                w_off_s = len(''.join([w for w in words[:ph_ids[0]]])) if ph_ids[0] != 0 else 0
                w_off_s += sum(blanks[:-1])
                w_off_e = w_off_s + len(ph) - 1
                phrases[ph]["char_offset"].append([w_off_s, w_off_e])
        return phrases

    def __call__(self, texts):
        return self.tokenize(texts)

    def test(self):
        texts = ["美国白宫国安会战略沟通协调员柯比（John Kirby）说，美中元首会晤仍在积极安排中。",
                 " 还会流传千古图啥呢老妖婆，呸！",
                 " 特朗普称，佩洛西去台湾是帮中国圆梦，是助力中国 一位川建国一位佩兴华统一祖国你俩功不可没",
                 "佩洛西无底线“驰名美式双标”遭到舆论长期讽刺"
                 ]
        res = self(texts)
        offsets = res["tok/offsets"]
        tokens = res["tok/fine"]
        words = [[text[s:e] for s, e in offs] for text, offs in zip(texts, offsets)]
        assert (words == tokens)
        print("test tokenize ok")
        res.pretty_print()  # 打印输出
        return res

if __name__ == "__main__":
    a = ["公安机关集中侦破一批涉渔涉砂类刑事案件，确保长江水域生态安全",
         "美国贸易代表发出照片显示，美国主管亚太地区贸易事务的副贸易代表莎拉∙比亚奇（右上）与台湾行政院政务委员兼经贸总谈判代表邓振中（左上）举行视频会晤。",
         "4月8日，全国打击治理电信网络新型违法犯罪工作电视电话会议在京召开。",
         "国务委员、公安部部长、国务院打击治理电信网络新型违法犯罪工作部际联席会议总召集人赵克志出席会议并讲话。",
        ]
    # a=["营前变110kV母联出线侧套管接口处过热处理"]
    h = PhraseExtractor()
    res0 = h.rule_based_phrase_extract(a, min_len=2)
    print(res0)
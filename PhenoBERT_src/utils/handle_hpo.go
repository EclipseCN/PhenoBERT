package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
)

var (
	obo_file_path  = ""
	save_file_path = ""
	HPOs           = map[string]*HPO_class{}
)

type HPO_class struct {
	Id      string
	Name    []string
	Alt_id  []string
	Def     []string
	Comment []string
	Synonym []string
	Xref    []string
	Is_a    []string
	Son     map[string]bool
	Father  map[string]bool
	Child   map[string]bool
}

// 用于产生HPO字典的文件
func main() {
	if len(os.Args) != 3 {
		fmt.Println("handle_hpo obo_file_path save_file_path")
		os.Exit(1)
	}
	loading(os.Args[1], os.Args[2])
}
func loading(obo_file_path string, save_file_path string) {
	fi, err := ioutil.ReadFile(obo_file_path)
	if err != nil {
		fmt.Println("Something wrong when open obo file")
		os.Exit(1)
	}
	obo_terms := strings.Split(string(fi), "[Term]")
	for obo_index := range obo_terms {
		term := obo_terms[obo_index]
		if strings.Contains(term, "id: ") {
			hpo := &HPO_class{"", make([]string, 0), make([]string, 0), make([]string, 0), make([]string, 0), make([]string, 0), make([]string, 0), make([]string, 0), make(map[string]bool), make(map[string]bool), make(map[string]bool)}
			seq := strings.Split(term, "\n")
			for term_index := range seq {
				// subSeq是每一条目行
				subSeq := seq[term_index]
				if subSeq == "" {
					continue
				}
				switch {
				case strings.HasPrefix(subSeq, "id: "):
					hpo.Id = strings.Split(subSeq, "id: ")[1]
				case strings.HasPrefix(subSeq, "name: "):
					hpo.Name = append(hpo.Name, strings.Split(subSeq, "name: ")[1])
				case strings.HasPrefix(subSeq, "alt_id: "):
					hpo.Alt_id = append(hpo.Alt_id, strings.Split(subSeq, "alt_id: ")[1])
				case strings.HasPrefix(subSeq, "def: "):
					hpo.Def = append(hpo.Def, strings.Split(subSeq, "\"")[1])
				case strings.HasPrefix(subSeq, "comment: "):
					hpo.Comment = append(hpo.Comment, strings.Split(subSeq, "comment: ")[1])
				case strings.HasPrefix(subSeq, "synonym: "):
					if strings.Contains(subSeq, "\"") {
						hpo.Synonym = append(hpo.Synonym, strings.Split(subSeq, "\"")[1])
					} else {
						hpo.Synonym = append(hpo.Synonym, strings.Split(subSeq, "synonym: ")[1])
					}
				case strings.HasPrefix(subSeq, "xref: "):
					hpo.Xref = append(hpo.Xref, strings.Split(subSeq, "xref: ")[1])
				case strings.HasPrefix(subSeq, "is_a: "):
					// 只加入HPO号
					hpo.Is_a = append(hpo.Is_a, strings.Replace(strings.Split(strings.Split(subSeq, "is_a: ")[1], "!")[0], " ", "", -1))
				}
			}
			HPOs[hpo.Id] = hpo
		}
	}
	for id := range HPOs {
		find_father(id, id) // 找爷爷
	}
	for id := range HPOs {
		for _, fa_hpo := range HPOs[id].Is_a {
			HPOs[fa_hpo].Son[id] = true
		}
	}
	for id := range HPOs {
		for fa_id := range HPOs[id].Father {
			if _, ok := HPOs[fa_id]; ok {
				HPOs[fa_id].Child[id] = true
			}
		}
	}
	save(HPOs, save_file_path)
}
func find_father(ori_id string, id string) {
	// 包括爷爷节点
	for _, fa_hpo := range HPOs[id].Is_a {
		HPOs[ori_id].Father[fa_hpo] = true
		if _, ok := HPOs[fa_hpo]; ok {
			find_father(ori_id, fa_hpo)
		}
	}
}
func save(HPOs map[string]*HPO_class, save_file_path string) {
	write_file, err := os.OpenFile(save_file_path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		fmt.Println("Can't open write file.")
		os.Exit(1)
	}
	defer write_file.Close()
	data, err := json.Marshal(HPOs)
	if err != nil {
		fmt.Println("Can't pickle the map.")
		os.Exit(1)
	}
	write_file.Write(data)
}

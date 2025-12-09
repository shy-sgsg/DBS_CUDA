#include <tinyxml2.h>
#include <string>
#include <iostream>
#include <cmath>
#include "DbsStitcher.hpp"

using namespace tinyxml2;

static const XMLElement *mustChild(const XMLElement *parent, const char *tag)
{
    if (!parent)
        return nullptr;
    if (const XMLElement *e = parent->FirstChildElement(tag))
        return e;
    return nullptr;
}
static bool getTextStr(const XMLElement *parent, const char *tag, std::string &out)
{
    if (const XMLElement *e = mustChild(parent, tag))
    {
        if (const char *t = e->GetText())
        {
            out = t;
            return true;
        }
    }
    return false;
}
static bool getTextDouble(const XMLElement *parent, const char *tag, double &out)
{
    if (const XMLElement *e = mustChild(parent, tag))
    {
        // 更稳健：QueryDoubleText 可忽略空白
        if (e->QueryDoubleText(&out) == XML_SUCCESS)
            return true;
        if (const char *t = e->GetText())
        {
            try
            {
                out = std::stod(t);
                return true;
            }
            catch (...)
            {
            }
        }
    }
    return false;
}
static bool getTextInt(const XMLElement *parent, const char *tag, int &out)
{
    if (const XMLElement *e = mustChild(parent, tag))
    {
        if (e->QueryIntText(&out) == XML_SUCCESS)
            return true;
        if (const char *t = e->GetText())
        {
            try
            {
                out = int(std::llround(std::stod(t)));
                return true;
            }
            catch (...)
            {
            }
        }
    }
    return false;
}

bool DbsStitcher::loadParamsFromXml(const std::string &xmlPath, Params &P)
{
    XMLDocument doc;
    XMLError err = doc.LoadFile(xmlPath.c_str());
    if (err != XML_SUCCESS)
    {
        std::cerr << "[XML] LoadFile failed: " << xmlPath
                  << " | tinyxml2 err=" << err
                  << " | msg=" << (doc.ErrorStr() ? doc.ErrorStr() : "(null)") << "\n";
        return false;
    }

    const XMLElement *root = doc.FirstChildElement("GMTI");
    if (!root)
    {
        std::cerr << "[XML] Missing <GMTI>\n";
        return false;
    }
    const XMLElement *prm = root->FirstChildElement("GMTI_parameter");
    if (!prm)
    {
        std::cerr << "[XML] Missing <GMTI_parameter>\n";
        return false;
    }

    // ---- 字符串必填 ----
    if (!getTextStr(prm, "GMTI_data2", P.dbs_data_path))
    {
        std::cerr << "[XML] Missing <GMTI_data>\n";
        return false;
    }
    if (!getTextStr(prm, "result_add", P.result_dir))
    {
        std::cerr << "[XML] Missing <result_add>\n";
        return false;
    }
    if (!getTextStr(prm, "Plane_POS", P.pos_path))
    {
        std::cerr << "[XML] Missing <Plane_POS>\n";
        return false;
    }
    if (!getTextStr(prm, "reffunc_add", P.reffunc_path))
    {
    }

    // ---- 整数/计数 ----
    if (!getTextInt(prm, "info_len", P.frame_header_len))
        return false;
    if (!getTextInt(prm, "pulse_len", P.range_samp_total))
        return false;
    if (!getTextInt(prm, "rg_len", P.range_samp_used))
        return false;
    if (!getTextInt(prm, "pulse_num", P.pulses_per_beam))
        return false;

    // ---- 频率/带宽/采样率/脉宽：单位换算（你的XML是 GHz/MHz/us）----
    {
        double fc_GHz = 0, Br_MHz = 0, fs_MHz = 0, Tr_us = 0;
        if (!getTextDouble(prm, "fc", fc_GHz))
            return false; // 17 -> GHz
        if (!getTextDouble(prm, "Br", Br_MHz))
            return false; // 50 -> MHz
        if (!getTextDouble(prm, "fs", fs_MHz))
            return false; // 75 -> MHz
        if (!getTextDouble(prm, "Tr", Tr_us))
            return false; // 2  -> us
        P.fc_hz = fc_GHz * 1e9;
        P.B_hz = Br_MHz * 1e6;
        P.fs_hz = fs_MHz * 1e6;
        P.tau_s = Tr_us * 1e-6;
    }

    if (!getTextDouble(prm, "PRF", P.PRF))
        return false; // 已是 Hz
    if (!getTextDouble(prm, "Rmin", P.Rmin_m))
        return false;

    // ---- 成像/扫描参数 ----
    if (!getTextInt(prm, "az_count", P.beams_per_period))
        return false;
    if (!getTextDouble(prm, "boshu", P.beamwidth_deg))
        return false;
    if (!getTextDouble(prm, "max_theta", P.scan_max_az_deg))
        return false;
    if (!getTextInt(prm, "period_first", P.period_first))
        return false;
    if (!getTextInt(prm, "period_num", P.period_count))
        return false;
    if (!getTextInt(prm, "n_tiaoguo", P.beam_skip))
        return false;
    if (!getTextDouble(prm, "ref_lon", P.lon_ref_deg))
        return false;
    if (!getTextDouble(prm, "raw_fenbianlv", P.out_res_m))
        return false;
    if (!getTextInt(prm, "len_tiaoguo", P.range_skip))
        return false;

    // ---- 可选项 ----
    (void)getTextInt(prm, "skip_pulses", P.time_skip_pulses);   // 91137
    (void)getTextDouble(prm, "ref_H", P.mean_ground_h);   // 92.435
    (void)getTextDouble(prm, "week_offset", P.gps_week_offset); // 0
    (void)getTextDouble(prm, "secBias", P.secBias);             // 44
    if (!getTextInt(prm, "isPC", P.isPC))
        return false;
    (void)getTextInt(prm, "hasRefFunc", P.hasRefFunc);   // 0/1
    (void)getTextInt(prm, "squint_side", P.squint_side); // 0/1

    // 没在XML的，可保留默认：interp_mode / flight_dir_flag

    // ---- 合法化 ----
    if (P.range_skip <= 0)
        P.range_skip = 1;
    if (P.range_samp_used > 0 && P.range_skip > P.range_samp_used)
    {
        P.range_skip = std::max(1, P.range_samp_used / 2);
    }

    // ---- 最小合理性检查 ----
    auto needPos = [&](double v, const char *name)
    {
        if (!(v > 0))
        {
            std::cerr << "[XML] invalid non-positive: " << name << "\n";
            return false;
        }
        return true;
    };
    if (!needPos(P.fc_hz, "fc") || !needPos(P.B_hz, "Br") || !needPos(P.fs_hz, "fs") ||
        !needPos(P.tau_s, "Tr") || !needPos(P.PRF, "PRF") || !needPos(P.beamwidth_deg, "boshu"))
        return false;

    return true;
}

// 非 const 版本的 mustChild（用于写）
static XMLElement *findChild(XMLElement *parent, const char *tag)
{
    if (!parent)
        return nullptr;
    return parent->FirstChildElement(tag);
}

// 确保子节点存在：若不存在且 createIfMissing=true，则自动创建
static XMLElement *ensureChild(XMLElement *parent,
                               const char *tag,
                               bool createIfMissing = false)
{
    if (!parent)
        return nullptr;
    XMLElement *e = parent->FirstChildElement(tag);
    if (!e && createIfMissing)
    {
        XMLDocument *doc = parent->GetDocument();
        if (!doc)
            return nullptr;
        e = doc->NewElement(tag);
        parent->InsertEndChild(e);
    }
    return e;
}

// 写 string：<tag>value</tag>
static bool setTextStr(XMLElement *parent,
                       const char *tag,
                       const std::string &value,
                       bool createIfMissing = false)
{
    XMLElement *e = ensureChild(parent, tag, createIfMissing);
    if (!e)
        return false;
    e->SetText(value.c_str());
    return true;
}

// 写 double：<tag>3.14</tag>
static bool setTextDouble(XMLElement *parent,
                          const char *tag,
                          double value,
                          bool createIfMissing = false)
{
    XMLElement *e = ensureChild(parent, tag, createIfMissing);
    if (!e)
        return false;
    e->SetText(value); // tinyxml2 会自己格式化成字符串
    return true;
}

// 写 int：<tag>123</tag>
static bool setTextInt(XMLElement *parent,
                       const char *tag,
                       int value,
                       bool createIfMissing = false)
{
    XMLElement *e = ensureChild(parent, tag, createIfMissing);
    if (!e)
        return false;
    e->SetText(value);
    return true;
}

bool DbsStitcher::saveParamsToXml(const std::string &xmlPath,
                                  const double &P)
{
    XMLDocument doc;
    XMLError err = doc.LoadFile(xmlPath.c_str());
    if (err != XML_SUCCESS)
    {
        std::cerr << "[XML] LoadFile failed: " << xmlPath << "\n";
        return false;
    }

    XMLElement *root = doc.FirstChildElement("GMTI");
    if (!root)
        return false;
    XMLElement *prm = root->FirstChildElement("GMTI_parameter");
    if (!prm)
        return false;

    // 例如更新 secBias / week_offset 等
    (void)setTextDouble(prm, "squint_angle", P, true);

    // 保存
    err = doc.SaveFile(xmlPath.c_str());
    if (err != XML_SUCCESS)
    {
        std::cerr << "[XML] SaveFile failed: " << xmlPath << "\n";
        return false;
    }
    return true;
}

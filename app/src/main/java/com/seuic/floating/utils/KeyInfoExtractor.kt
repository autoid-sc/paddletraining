package com.seuic.floating.utils

import android.util.Log
import java.util.regex.Pattern

/**
 * 关键信息提取器
 * 用于从OCR识别结果中提取商品、票据、铭牌等关键信息
 */
class KeyInfoExtractor {

    companion object {
        private const val TAG = "KeyInfoExtractor"
    }

    /**
     * 识别类型
     */
    enum class RecognitionType {
        PRODUCT,    // 商品包装
        TICKET,     // 票据
        PLATE,      // 铭牌
        GENERAL     // 通用
    }

    /**
     * 商品信息
     */
    data class ProductInfo(
        var name: String = "",           // 商品名称
        var specification: String = "",  // 规格
        var batchNumber: String = "",    // 批号
        var expiryDate: String = "",     // 有效期
        var manufacturer: String = ""    // 生产厂家
    )

    /**
     * 票据信息
     */
    data class TicketInfo(
        var amount: String = "",         // 金额
        var date: String = "",           // 日期
        var title: String = "",          // 抬头/公司名称
        var invoiceNumber: String = ""   // 票据号码
    )

    /**
     * 铭牌信息
     */
    data class PlateInfo(
        var model: String = "",          // 型号
        var serialNumber: String = "",   // 序列号
        var productionDate: String = "", // 出厂日期
        var manufacturer: String = "",   // 生产厂家
        var power: String = "",          // 功率
        var voltage: String = ""         // 电压
    )

    /**
     * 提取关键信息
     * @param ocrText OCR识别的文本
     * @param type 识别类型
     * @return 格式化后的JSON字符串
     */
    fun extractKeyInfo(ocrText: String, type: RecognitionType): String {
        return when (type) {
            RecognitionType.PRODUCT -> extractProductInfo(ocrText).toJson()
            RecognitionType.TICKET -> extractTicketInfo(ocrText).toJson()
            RecognitionType.PLATE -> extractPlateInfo(ocrText).toJson()
            RecognitionType.GENERAL -> ocrText
        }
    }

    /**
     * 提取商品信息
     */
    private fun extractProductInfo(text: String): ProductInfo {
        val info = ProductInfo()

        // 提取商品名称（通常在文本开头，或包含"品名"、"产品"等关键词）
        info.name = extractProductName(text)

        // 提取规格（常见格式：500ml、100g、10×10等）
        info.specification = extractSpecification(text)

        // 提取批号（常见格式：批号、Lot、Batch等）
        info.batchNumber = extractBatchNumber(text)

        // 提取有效期（常见格式：有效期、Exp Date、生产日期+保质期等）
        info.expiryDate = extractExpiryDate(text)

        // 提取生产厂家
        info.manufacturer = extractManufacturer(text)

        return info
    }

    /**
     * 提取票据信息
     */
    private fun extractTicketInfo(text: String): TicketInfo {
        val info = TicketInfo()

        // 提取金额（常见格式：￥、¥、$、元、金额、合计等）
        info.amount = extractAmount(text)

        // 提取日期（常见格式：YYYY-MM-DD、YYYY/MM/DD、YYYY年MM月DD日等）
        info.date = extractDate(text)

        // 提取抬头/公司名称（通常在票据顶部）
        info.title = extractTitle(text)

        // 提取票据号码（常见格式：No.、编号、号码等）
        info.invoiceNumber = extractInvoiceNumber(text)

        return info
    }

    /**
     * 提取铭牌信息
     */
    private fun extractPlateInfo(text: String): PlateInfo {
        val info = PlateInfo()

        // 提取型号（常见格式：型号、Model、Type等）
        info.model = extractModel(text)

        // 提取序列号（常见格式：SN、S/N、序列号、Serial No.等）
        info.serialNumber = extractSerialNumber(text)

        // 提取出厂日期
        info.productionDate = extractProductionDate(text)

        // 提取生产厂家
        info.manufacturer = extractManufacturer(text)

        // 提取功率（常见格式：功率、Power、W、kW等）
        info.power = extractPower(text)

        // 提取电压（常见格式：电压、Voltage、V等）
        info.voltage = extractVoltage(text)

        return info
    }

    // ==================== 商品信息提取方法 ====================

    private fun extractProductName(text: String): String {
        val patterns = listOf(
            Pattern.compile("品名[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("产品名称[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("商品[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }

        // 如果没有找到明确的品名字段，取第一行或前几个字
        val lines = text.split("\n")
        for (line in lines) {
            if (line.trim().length > 2 && !line.contains("批次") && !line.contains("有效期")) {
                return line.trim()
            }
        }
        return ""
    }

    private fun extractSpecification(text: String): String {
        val patterns = listOf(
            Pattern.compile("规格[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("净含量[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("(\\d+\\.?\\d*\\s*(ml|g|kg|L|升|克|毫升|千克|片|粒|包|盒))", Pattern.CASE_INSENSITIVE),
            Pattern.compile("(\\d+×\\d+)") // 10×10等格式
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return if (matcher.groupCount() >= 1) {
                    matcher.group(1)?.trim() ?: ""
                } else {
                    matcher.group()?.trim() ?: ""
                }
            }
        }
        return ""
    }

    private fun extractBatchNumber(text: String): String {
        val patterns = listOf(
            Pattern.compile("批号[：:：]\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Lot[：:：]?[Nn]o?[：:：]?\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Batch[：:：]?[Nn]o?[：:：]?\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("生产批号[：:：]\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    private fun extractExpiryDate(text: String): String {
        val patterns = listOf(
            Pattern.compile("有效期[至到]?[：:：]\\s*(\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2}[日]?)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Exp[：:：]?[\\sDd][aA][tT][eE][：:：]?\\s*(\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2})", Pattern.CASE_INSENSITIVE),
            Pattern.compile("保质期[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("有效期至\\s*(\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2})")
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    // ==================== 票据信息提取方法 ====================

    private fun extractAmount(text: String): String {
        val patterns = listOf(
            Pattern.compile("金额[：:：]\\s*([￥¥$]?\\s*\\d+[.,]\\d{2})", Pattern.CASE_INSENSITIVE),
            Pattern.compile("合计[：:：]\\s*([￥¥$]?\\s*\\d+[.,]\\d{2})", Pattern.CASE_INSENSITIVE),
            Pattern.compile("总计[：:：]\\s*([￥¥$]?\\s*\\d+[.,]\\d{2})", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Total[：:：]?\\s*([￥¥$]?\\s*\\d+[.,]\\d{2})", Pattern.CASE_INSENSITIVE),
            Pattern.compile("([￥¥$]\\s*\\d+[.,]\\d{2})")
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    private fun extractDate(text: String): String {
        val patterns = listOf(
            Pattern.compile("日期[：:：]\\s*(\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2}[日]?)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("时间[：:：]\\s*(\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2})", Pattern.CASE_INSENSITIVE),
            Pattern.compile("\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2}[日]?") // 匹配任意日期格式
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return if (matcher.groupCount() >= 1) {
                    matcher.group(1)?.trim() ?: matcher.group()?.trim() ?: ""
                } else {
                    matcher.group()?.trim() ?: ""
                }
            }
        }
        return ""
    }

    private fun extractTitle(text: String): String {
        val lines = text.split("\n")
        for (line in lines) {
            val trimmed = line.trim()
            // 假设标题在票据顶部，通常包含"公司"、"有限"、"发票"、"收据"等关键词
            if (trimmed.contains("公司") || trimmed.contains("有限") ||
                trimmed.contains("发票") || trimmed.contains("收据")) {
                return trimmed
            }
        }
        return ""
    }

    private fun extractInvoiceNumber(text: String): String {
        val patterns = listOf(
            Pattern.compile("号码[：:：]\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("No[：:：]?\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("编号[：:：]\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    // ==================== 铭牌信息提取方法 ====================

    private fun extractModel(text: String): String {
        val patterns = listOf(
            Pattern.compile("型号[：:：]\\s*([A-Za-z0-9-_]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Model[：:：]?\\s*([A-Za-z0-9-_]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Type[：:：]?\\s*([A-Za-z0-9-_]+)", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    private fun extractSerialNumber(text: String): String {
        val patterns = listOf(
            Pattern.compile("SN[：:：]?\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("S/N[：:：]?\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("序列号[：:：]\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Serial[：:：]?[Nn]o?[：:：]?\\s*([A-Za-z0-9]+)", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    private fun extractProductionDate(text: String): String {
        val patterns = listOf(
            Pattern.compile("出厂日期[：:：]\\s*(\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2})", Pattern.CASE_INSENSITIVE),
            Pattern.compile("生产日期[：:：]\\s*(\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2})", Pattern.CASE_INSENSITIVE),
            Pattern.compile("MFG[：:：]?[Dd][aA][tT][eE][：:：]?\\s*(\\d{4}[-/年]\\d{1,2}[-/月]\\d{1,2})", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    private fun extractPower(text: String): String {
        val patterns = listOf(
            Pattern.compile("功率[：:：]\\s*(\\d+\\.?\\d*\\s*(W|kW|瓦|千瓦))", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Power[：:：]?\\s*(\\d+\\.?\\d*\\s*(W|kW))", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    private fun extractVoltage(text: String): String {
        val patterns = listOf(
            Pattern.compile("电压[：:：]\\s*(\\d+\\.?\\d*\\s*(V|伏))", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Voltage[：:：]?\\s*(\\d+\\.?\\d*\\s*(V))", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    private fun extractManufacturer(text: String): String {
        val patterns = listOf(
            Pattern.compile("生产厂家[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("制造商[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("生产商[：:：]\\s*(.+)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("Manufacturer[：:：]?\\s*(.+)", Pattern.CASE_INSENSITIVE)
        )

        for (pattern in patterns) {
            val matcher = pattern.matcher(text)
            if (matcher.find()) {
                return matcher.group(1)?.trim() ?: ""
            }
        }
        return ""
    }

    // ==================== 辅助方法 ====================

    /**
     * 将ProductInfo转换为JSON字符串
     */
    private fun ProductInfo.toJson(): String {
        return org.json.JSONObject().apply {
            put("type", "商品")
            put("name", name)
            put("specification", specification)
            put("batchNumber", batchNumber)
            put("expiryDate", expiryDate)
            put("manufacturer", manufacturer)
        }.toString()
    }

    /**
     * 将TicketInfo转换为JSON字符串
     */
    private fun TicketInfo.toJson(): String {
        return org.json.JSONObject().apply {
            put("type", "票据")
            put("amount", amount)
            put("date", date)
            put("title", title)
            put("invoiceNumber", invoiceNumber)
        }.toString()
    }

    /**
     * 将PlateInfo转换为JSON字符串
     */
    private fun PlateInfo.toJson(): String {
        return org.json.JSONObject().apply {
            put("type", "铭牌")
            put("model", model)
            put("serialNumber", serialNumber)
            put("productionDate", productionDate)
            put("manufacturer", manufacturer)
            put("power", power)
            put("voltage", voltage)
        }.toString()
    }
}
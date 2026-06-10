# TỪ ĐIỂN DỮ LIỆU — Debt Collection Intelligence Platform

> **Phiên bản**: 2.0  
> **Cập nhật lần cuối**: 2026-06-05  
> **Tệp nguồn gốc**: `TONG_HOP_NAM_2026.xlsx` (Sheet: `DOANH SỐ`)  
> **Cột gốc Excel**: 101 cột  
> **Cột sau xử lý Step 1** (bỏ PII): ~76 cột tĩnh + các cột động theo tháng  
> **Người cập nhật định nghĩa**: *(chưa điền — chờ review từ phụ trách)*

---

## Quy ước ký hiệu

| Ký hiệu | Ý nghĩa |
|---------|---------|
| 🔴 **PII** | Cột chứa thông tin cá nhân nhạy cảm — **BỊ XÓA HOÀN TOÀN tại Step 1** |
| 🟢 **STATIC** | Cột tĩnh: giá trị không thay đổi theo tháng (hợp đồng, nhân thân, dư nợ snapshot) |
| 🔵 **DYNAMIC** | Cột động: mỗi tháng có một bản ghi — sau khi user chuẩn hóa, **prefix tháng bị xóa**, cột `THÁNG` đóng vai trò nhận diện thời gian |

---

## PHẦN 1 — CỘT PII (BỊ DROP TẠI STEP 1)

> Các cột này tồn tại trong file Excel gốc nhưng sẽ bị xóa ngay trong bước đầu tiên (`excel_to_csv.py`) trước khi lưu ra CSV. **Không bao giờ được lưu vào bất kỳ file nào trong pipeline.**

| STT | Tên Cột Gốc (Excel) | Loại | Định Nghĩa / Ý Nghĩa |
|-----|---------------------|------|----------------------|
| 1 | `NAME` | 🔴 PII | Họ và tên của chủ hồ sơ vay |
| 2 | `CMND` | 🔴 PII | Số định danh cá nhân của chủ hồ sơ vay |
| 3 | `SỐ ĐIỆN THOẠI` | 🔴 PII | Số điện thoại của chủ hồ sơ vay |
| 4 | `MAIL` | 🔴 PII | Email của chủ hồ sơ vay |
| 5 | `TÊN CÔNG TY` | 🔴 PII | Tên công ty chủ hồ sơ vay đang làm việc |
| 6 | `ĐỊA CHỈ CÔNG TY` | 🔴 PII | Địa chỉ công ty chủ hồ sơ vay đang làm việc |
| 7 | `SĐT CÔNG TY` | 🔴 PII | Số diện thoại công ty chủ hồ sơ vay đang làm việc |
| 8 | `THAM CHIẾU VỢ/CHỒNG` | 🔴 PII | Thông tin tên vợ/chồng chủ hồ sơ vay |
| 9 | `SỐ THAM CHIẾU 1` | 🔴 PII | Số điện thoại của tham chiếu cột QUAN HỆ 1 |
| 10 | `QUAN HỆ 1` | 🔴 PII | Thông tin tên người thứ nhất có quan hệ với chủ hồ sơ vay |
| 11 | `THAM CHIẾU 2` | 🔴 PII | Số điện thoại của tham chiếu cột QUAN HỆ 2 |
| 12 | `QUAN HỆ 2` | 🔴 PII | Thông tin tên người thứ hai có quan hệ với chủ hồ sơ vay |
| 13 | `THAM CHIẾU 3` | 🔴 PII | Số điện thoại của tham chiếu cột QUAN HỆ 3 |
| 14 | `QUAN HỆ 3` | 🔴 PII | Thông tin tên người thứ ba có quan hệ với chủ hồ sơ vay |
| 15 | `THAM CHIẾU 4` | 🔴 PII | Số điện thoại của tham chiếu cột QUAN HỆ 4 |
| 16 | `QUAN HỆ 4` | 🔴 PII | Thông tin tên người thứ tư có quan hệ với chủ hồ sơ vay |
| 17 | `TÁC ĐỘNG CŨ` | 🔴 PII | Chứa các số điện thoại đã từng tác động có liên qua tới chủ hồ sơ vay |
| 18 | `STK BIDV` | 🔴 PII | Số tài khoản ngân hàng BIDV |
| 19 | `STK WORRI BANK` | 🔴 PII | Số tài khoản ngân hàng Worri Bank |
| 20 | `NƠI NHẬN THƯ` | 🔴 PII | Công ty nơi chủ hồ sơ vay làm việc |
| 21 | `SĐT` | 🔴 PII | Số diện thoại công ty chủ hồ sơ vay đang làm việc |
| 22 | `ĐDPL` | 🔴 PII | Người đại diện pháp lý để tranh tụng với chủ hồ sơ vay |
| 23 | `NGÀY GỬI` | 🔴 PII | Ngày gửi thư thông báo |
| 24 | `MÃ VẬN ĐƠN` | 🔴 PII | Mã vận đơn thư thông báo |
| 25 | `HOTLINE` | 🔴 PII | Số hotline - hiện tại chưa thấy được sử dụng |

---

## PHẦN 2 — CỘT TĨNH (STATIC COLUMNS)

> Các cột này có giá trị **cố định theo hợp đồng/khách hàng**, không thay đổi theo tháng. Đây là nền tảng cho Feature Engineering và mô hình phân tích.

### Nhóm A — Định Danh & Nhân Thân Khách Hàng

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 1 | `LOAN ID` | string | `112401227903075`, `542301065665283` | Mã hợp đồng/hồ sơ vay |
| 2 | `GIỚI TÍNH` | category | `Bà`, `Ông`, 'Nam', 'Nữ' | Giới tính chủ hồ sơ vay |
| 3 | `NGÀY THÁNG NĂM SINH` | datetime | Excel serial → DD/MM/YYYY | Ngày/tháng/năm sinh chủ hồ sơ vay |
| 4 | `TUỔI` | int64 | `32`, `28`, `25` | Tuổi chủ hồ sơ vay |
| 5 | `CCCD/CMND` | category | `CCCD`, `CMND` | Giấy tờ định danh chủ hồ sơ vay sử dụng CCCD hoặc CMND |

### Nhóm B — Địa Chỉ Khách Hàng

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 6 | `ĐỊA CHỈ TẠM TRÚ` | string (text) | `486 DƯƠNG THỊ MƯỜI, PHƯỜNG TÂN THỚI HIỆP, Q12, TP.HCM` | Địa chỉ tạm trú chủ hồ sơ vay |
| 7 | `PHƯỜNG TẠM TRÚ` | category | `PHƯỜNG TÂN THỚI HIỆP`, `XÃ CAM HIẾU` | Phường/xã tạm trú chủ hồ sơ vay |
| 8 | `QUẬN/HUYỆN TẠM TRÚ` | category | `CAM LỘ`, `CHƠN THÀNH`, `12` | Quận/huyện tạm trú chủ hồ sơ vay |
| 9 | `TỈNH TẠM TRÚ` | category | `TỈNH QUẢNG TRỊ`, `TP. HỒ CHÍ MINH` | Tỉnh tạm trú chủ hồ sơ vay |
| 10 | `ĐỊA CHỈ THƯỜNG TRÚ` | string (text) | `ẤP 2, XÃ VĨNH XƯƠNG, THỊ XÃ TÂN CHÂU, AN GIANG` | Địa chỉ thường trú chủ hồ sơ vay |
| 11 | `PHƯỜNG THƯỜNG TRÚ` | category | `XÃ HIỆP HÒA`, `XÃ LONG SƠN` | Phường/xã thường trú chủ hồ sơ vay |
| 12 | `QUẬN/HUYỆN THƯỜNG TRÚ` | category | `HIỆP ĐỨC`, `CẦU NGANG`, `TÂN CHÂU` | Quận/huyện thường trú chủ hồ sơ vay |
| 13 | `TỈNH THƯỜNG TRÚ` | category | `TỈNH QUẢNG NAM`, `TỈNH TRÀ VINH` | Tỉnh thường trú chủ hồ sơ vay |

### Nhóm C — Nghề Nghiệp & Thu Nhập

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 14 | `MỨC LƯƠNG` | float64 | `40,000,000`, `8,500,000`, `0` | Mức thu nhập hàng tháng chủ hồ sơ vay |
| 15 | `CHỨC VỤ` | category | `CÔNG NHÂN`, `TỰ DOANH`, `KHÁC` | Vị trí, chức vụ của chủ hồ sơ vay |
| 16 | `PL NGHÀNH NGHỀ` | category | `CÔNG NGHIỆP CHẾ BIẾN`, `XÂY DỰNG`, `DỊCH VỤ KHÁC` | Ngành nghề của chủ hồ sơ vay |
| 17 | `SỐ ĐIỆN THOẠI.1` | string | `0385724865`, `0000000000` | Số điện thoại của cột THAM CHIẾU VỢ/CHỒNG |

### Nhóm D — Thông Tin Hợp Đồng Vay

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 18 | `NGÀY GIẢI NGÂN` | datetime | Excel serial → DD/MM/YYYY | Ngày khoản giải ngân |
| 19 | `SỐ TIỀN GIẢI NGÂN` | float64 | `43,000,000`, `7,105,000` | Số tiền giải ngân |
| 20 | `LÃI SUẤT` | float64 | `54.0`, `57.0`, `0.0` | Lãi suất – đơn vị: %/năm |
| 21 | `SỐ KỲ` | int64 | `36`, `6`, `3` | Số kỳ – đơn vị: tháng |
| 22 | `PHÍ BẢO HIỂM` | float64 | `3,000,000`, `2,025,000` | Phí bảo hiểm khoản vay |
| 23 | `SỐ TIỀN THANH TOÁN HÀNG THÁNG` | float64 | `2,434,928`, `1,700,378` | Số tiền thanh toán hàng tháng/kỳ |
| 24 | `NGÀY KẾT THÚC HĐ` | datetime | Excel serial → DD/MM/YYYY | Ngày kết thúc hợp đồng vay |
| 25 | `SẢN PHẨM` | category | `LOAN_PURPOSE_VEHICLE`, `LOAN_PURPOSE_SHOPPING`, `LOAN_PURPOSE_HOUSE` | Mục đích vay vốn |
| 26 | `HỢP ĐỒNG SỐ 2` | string (ID) | `112309206903471` | Mã hợp đồng thứ 2 nếu KH có nhiều hơn 1 khoản vay |
| 27 | `SỐ LƯỢNG HỢP ĐỒNG` | string | `1 HỢP ĐỒNG`, `2 HỢP ĐỒNG`, `3 HỢP ĐỒNG` | Số lượng hợp đồng vay|

### Nhóm E — Thông Tin Trả Nợ & Dư Nợ

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 28 | `SỐ KỲ ĐÃ TT` | int64 | `25`, `20`, `0` | Số tháng/kỳ đã thanh toán |
| 29 | `TỔNG ĐÃ THANH TOÁN` | float64 | `61,903,000`, `34,008,000`, `0` | Tổng số tiền đã thanh toán |
| 30 | `SỐ TIỀN TT GẦN NHẤT` | float64 | `39,988,000`, `1,687,000`, `0` | Số tiền thanh toán gần nhất |
| 31 | `NGÀY TT GẦN NHẤT` | datetime | Excel serial → DD/MM/YYYY | Ngày thanh toán gần nhất |
| 32 | `DPD` | int64 | `-22`, `171`, `1060` | Số ngày quá hạn (Days Past Due); giá trị âm = chưa đến hạn |
| 33 | `NỢ GỐC` | float64 | `23,580,138`, `18,758,633` | Số tiền gốc còn lại |
| 34 | `LÃI TRONG HẠN` | float64 | `0`, `4,750,330`, `4,454,906` | Tiền lãi trong hạn |
| 35 | `LÃI QUÁ HẠN` | float64 | `1,287,867`, `0`, `43,227,562` | Tiền lãi quá hạn |
| 36 | `PHÍ PHẠT` | float64 | *(cột trống — empty)* | Phí phạt do trễ hạn thanh toán |
| 37 | `TỔNG NỢ` | float64 | `23,580,138`, `24,796,830` | Tổng nợ của hồ sơ vay |
| 38 | `SỐ TIỀN THÔNG BÁO` | float64 | `23,580,138`, `24,152,896` | Số tiền thương lượng để cắt gốc hoặc giảm lãi giúp chủ hồ sơ vay thanh toán toàn bộ khoản nợ |

### Nhóm F — Phân Loại & Phân Khúc (Pre-computed)

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 39 | `VNE LAW PL 01` | category | `NEW`, `KEEP` | Tình trạng của hồ sơ vay: mới, giữ lại, kho của công ty Luật VNE |
| 40 | `VNE LAW PL 02` | category | `ROTATE.01.26`, `ROTATE.12.25`, `ROTATE.10.25` | Tháng đã hoặc đang tìm cách xử lý hồ sơ |
| 41 | `PHÂN LOẠI VÙNG MIỀN` | category | `SOUTH`, `MEKONG`, `CENTRAL`, `CENTRAL_HIGHLAND`, `NORTH` | Phân loại vùng miền: Miền Bắc (North), Miền Trung (Central), Vùng Tây Nguyên (Central Highland), Miền Nam (South) và Miền Tây (Mekong)  |
| 42 | `PHÂN LOẠI POS` | ordinal | `HIGH POS`, `MEDIUM POS`, `LOW POS` | Phân khúc theo mức dư nợ gốc |
| 43 | `PL NHÓM TUỔI` | ordinal | `<25`, `25-45`, `46-60`, `>60` | Phân loại nhóm tuổi chủ hồ sơ vay |
| 44 | `NHÓM DPD` | ordinal | `<360`, `<1000`, `<1800`, `>1800` | Nhóm quá hạn |
| 45 | `NHÓM SỐ LẦN TT` | ordinal | `0-3 LẦN`, `3-12 LẦN`, `13-24 LẦN`, `TRÊN 24 LẦN` | Phân loại nhóm số lần thanh toán |
| 46 | `ĐÁNH GIÁ KHÁCH HÀNG` | category | `A`, `B`, `D` | Đánh giá mức độ rủi ro của khách hàng |

### Nhóm G — Thông Tin Đối Tác & Dự Án

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 47 | `DỰ ÁN` | category | `SHB`, `HANMIR - LOTTE`, `LOTTE`, `MSB`, `ABBANK`, `MC`, `SVFC`, `BDI - SHB`, `HANMIR - MIRA` | **CỘT QUAN TRỌNG NHẤT**: Các đối tác của công ty Luật VNE |
| 48 | `ĐỐI TÁC PL1` | category | `NEW`, `KEEP` | Trạng thái hồ sơ vay đối tác gửi: mới, giữ lại |
| 49 | `ĐỐI TÁC PL2` | category | `ROTATE.01.26`, `ROTATE.12.25`, `ROTATE.11.25` | Tháng đã hoặc đang tìm cách xử lý hồ sơ đối tác gửi |
| 50 | `KHÁCH HÀNG NHIỀU DỰ ÁN` | category | `0`, `SHB_SVFC`, `SHB_HANMIR-LOTTE` | Khoản vay khác của chủ hồ sơ vay, có thể có nhiều hơn 1 dự án |
| 51 | `SỐ HD DÀI HANMIR` | string | `0`, `0001SA48U0014438`, `902001378972` | Số hợp đồng dài của đối tác Hanmir |

### Nhóm H — Mục Tiêu Hoạt Động (Tháng Hiện Tại)

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 52 | `SỐ CUỘC GỌI` | float64 | `4326`, `3528` | Số cuộc gọi trong tháng |
| 53 | `SỐ PHÚT GỌI` | float64 | `781.2`, `684.3` | Số phút gọi trong tháng |
| 54 | `MỤC TIÊU VNE` | float64 | `100,000,000`, `78,000,000` | KPI/Số tiền thu hồi mà VNE đặt ra trong tháng |
| 55 | `MỤC TIÊU ĐỐI TÁC` | float64 | `143,838`, `114,427` | KPI/Số tiền thu hồi mà đối tác đặt ra trong tháng |

### Nhóm I — Pháp Lý & Khởi Kiện

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 56 | `HỒ SƠ KHỞI KIỆN` | bool/category | `0`, `TỐ TỤNG` | Tình trạng hồ sơ vay có đang khởi kiện hay không |
| 57 | `THÁNG ĐỀ XUẤT KHỞI KIỆN` | category | `0`, `THÁNG 11.25`, `THÁNG 10.25` | Tháng đề xuất khởi kiện |

---

## PHẦN 3 — CỘT ĐỘNG (DYNAMIC COLUMNS — THEO THÁNG)

| STT | Tên Cột (Sau Chuẩn Hóa) | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|------------------------|-------|---------------|----------------------|
| 58 | `DỰ KIẾN KẾT QUẢ` | float64 | `0.0` | Số tiền dự kiến thu hồi được trong tháng |
| 59 | `NGÀY DỰ KIẾN CÓ KẾT QUẢ` | datetime | *(cột trống — empty)* | Ngày dự kiến thu hồi được tiền |
| 60 | `KẾT QUẢ` | float64 | `26,000,000`, `2,000,000`, `488,000` | Số tiền thực tế thu hồi được trong tháng |
| 61 | `NGÀY CÓ KẾT QUẢ` | datetime | Excel serial → DD/MM/YYYY | Ngày thực tế thu hồi được tiền |
| 62 | `MÃ TÌNH TRẠNG LIÊN HỆ` | string (ID) | `CBACK`, `NCON`, `PTP`, `FAIL` | Mã trạng thái liên hệ, ví dụ: `CBACK` (call back), `NCON` (no contact), `PTP` (promise to pay), `FAIL` (failed) |
| 63 | `TÌNH TRẠNG LIÊN HỆ` | string (text) | *(ghi chú tự do của collector)* | Ghi chú tự do của collector về tình trạng liên hệ |
| 64 | `TÌNH TRẠNG SMS` | category | `0`, `SMS - MẪU 01`, `SMS - Mẫu 01` | Trạng thái tin nhắn SMS, ví dụ: `0` (chưa gửi), `SMS - MẪU 01` (đã gửi mẫu 1), `SMS - MẪU 02` (đã gửi mẫu 2) |
| 65 | `NGÀY LÀM VIỆC` | datetime | Excel serial → DD/MM/YYYY | Ngày collector làm việc trực tiếp trên hồ sơ |
| 66 | `MÃ BHXH KHÁCH HÀNG` | string (ID) | `CHƯA XÁC ĐỊNH`, `DN4758421728840` | Mã BHXH để truy vết tình trạng việc làm |
| 67 | `TÌNH TRẠNG MÃ BHYT` | string (text) | `CHƯA XÁC ĐỊNH`, *(chuỗi dài từ API BHYT)* | Thông tin bảo hiểm y tế |
| 68 | `PHÂN LOẠI VL THEO MÃ` | string (ID) | `CHƯA XÁC ĐỊNH`, `CÒN HẠN`, `CHUYỂN` | Phân loại tình trạng việc làm |
| 69 | `TÌNH TRẠNG VL` | category | `CHƯA XÁC ĐỊNH`, `LÀM TẠI DOANH NGHIỆP`, `LAO ĐỘNG TỰ DO` | Tình trạng việc làm |
| 70 | `MÃ ĐKKCBBD` | string (ID) | `CHƯA XÁC ĐỊNH`, `70083`, `79488` | Mã nơi đăng ký khám chữa bệnh ban đầu |
| 71 | `NƠI ĐKKCBBD` | string (text) | `CHƯA XÁC ĐỊNH`, `Bệnh viện đa khoa Xuyên á - Củ Chi...` | Tên bệnh viện/cơ sở y tế đăng ký BHYT ban đầu |
| 72 | `PHỤ TRÁCH HỒ SƠ` | category | `Dungtq`, `Phucdn`, `KHO` | Tên collector phụ trách xử lý hồ sơ vay |
| 73 | `LEAD QUẢN LÝ HỒ SƠ` | category | `Tridt`, `KHO`, `CTV` | Tên/nhóm nhân viên quản lý hồ sơ vay |
| 74 | `CHI NHÁNH` | category | `HỒ CHÍ MINH`, `KHO`, `GĐ` | Khu vực nơi quản lý hồ sơ vay |
| 75 | `ĐỊA CHỈ` | string (text) | `275A/10, tổ 4A, khu phố Tân Lập, Đồng Nai` | Địa chỉ của công ty tương ứng với cột NƠI NHẬN THƯ |

---

## PHẦN 4 — CỘT ĐẶC BIỆT

| STT | Tên Cột | Dtype | Sample Values | Định Nghĩa / Ý Nghĩa |
|-----|---------|-------|---------------|----------------------|
| 76 | `THÁNG` | category | `THÁNG 01.26`, `THÁNG 02.26`, `THÁNG 03.26`... | **CỘT KHÓA THỜI GIAN**: định danh tháng của bản ghi. Sau khi user chuẩn hóa file Excel, đây là cột duy nhất phân biệt dữ liệu theo tháng |

---

## PHẦN 5 — BẢNG TỔNG HỢP NHANH (Quick Reference)

| Nhóm | Số Cột | Vai Trò Trong Pipeline |
|------|--------|------------------------|
| 🔴 PII (bị drop tại Step 1) | 25 | Xóa hoàn toàn, không đưa vào CSV |
| 🟢 Static — Định danh & Nhân thân | 5 | Join key, nhân thân KH |
| 🟢 Static — Địa chỉ | 8 | Geo analysis (Step 8D) |
| 🟢 Static — Nghề nghiệp & Thu nhập | 4 | Feature: DTI, income segment |
| 🟢 Static — Hợp đồng | 10 | Feature: loan_age, installment, product |
| 🟢 Static — Trả nợ & Dư nợ | 11 | Feature: repayment_ratio, DPD, EAD |
| 🟢 Static — Phân loại & Phân khúc | 8 | Segment, cluster baseline |
| 🟢 Static — Đối tác & Dự án | 5 | Partner filter (Step 6, 8H) |
| 🟢 Static — Mục tiêu | 4 | KPI tracking |
| 🟢 Static — Pháp lý | 2 | Litigation model (Step 8F) |
| 🔵 Dynamic — Kết quả thu hồi | 2 | **Target variable**: KẾT QUẢ |
| 🔵 Dynamic — Liên lạc & Collector | 13 | Contact funnel (Step 8B), Agent perf (Step 8F) |
| 🔵 Dynamic — BHXH/Việc làm | 6 | Employment signal for scoring |
| ⭐ Đặc biệt — Khóa thời gian | 1 | `THÁNG` — Time dimension |
| **TỔNG** | **~101** | |

---

> [!IMPORTANT]
> **Hướng dẫn cập nhật từ điển này:**
> 1. Mọi cột có ghi `*(Chờ định nghĩa)*` → người phụ trách cần điền nội dung vào cột **Định Nghĩa / Ý Nghĩa**
> 2. Nếu có cột mới phát sinh khi tháng mới được thêm vào, chỉ cần cập nhật **Phần 3** (Cột động) — tên cột không đổi
> 3. **Không được sửa tên cột** trong bảng này nếu chưa đồng bộ với `config.py`

---

*Tài liệu này được sinh ra từ `cols_excel.txt` (101 cột gốc), `refined_schema_ai.csv` và xác nhận thực tế từ phụ trách dữ liệu.*

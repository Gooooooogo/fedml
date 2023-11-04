import torch
def average(list):
    transposed = zip(* list)
    averages = [sum(column) / len(column) for column in transposed]
    return averages

def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


import openpyxl
def cleanexcel(excel_name, sheet_name, data):
    workbook = openpyxl.load_workbook(excel_name)

    # 指定要选择的工作表名称

    # 检查工作表是否存在，如果不存在则创建一个新工作表
    if sheet_name not in workbook.sheetnames:
        workbook.create_sheet(title=sheet_name)

    # 选择工作表
    sheet = workbook[sheet_name]

    # 删除工作表中的所有旧数据
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, max_col=2):
        for cell in row:
            cell.value = None
        column=1
    # 将数据写入选定的工作表，每次一行
    for key, _ in data.items():
        sheet.cell(row=1, column=column, value=key)  # 写入值
        column+= 1
    workbook.save(excel_name)
    workbook.close()

def save2excel(excel_name, sheet_name, data):
    workbook = openpyxl.load_workbook(excel_name)
    if sheet_name not in workbook.sheetnames:
        workbook.create_sheet(title=sheet_name)
    sheet = workbook[sheet_name]

    # # 删除工作表中的所有旧数据
    # for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, max_col=2):
    #     for cell in row:
    #         cell.value = None

    # 获取工作表中的最后一行
    last_row = sheet.max_row + 1
    '''
        data = {
        "current_global_round": current_global_round,
        "current_iteration": current_iteration,
        "average_training_loss": average_training_loss,
        "test_loss": test_loss,
        "accuracy": accuracy
        }
    '''
    column=1
    # 将数据写入选定的工作表，每次一行
    for key, value in data.items():
        sheet.cell(row=last_row, column=column, value=value)  # 写入值
        column+= 1
    # 保存Excel文件
    workbook.save(excel_name)

    # 关闭工作簿
    workbook.close()

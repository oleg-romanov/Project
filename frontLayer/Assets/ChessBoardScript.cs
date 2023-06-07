using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ChessBoardScript : MonoBehaviour
{
    // Ссылка на префаб клетки
    public GameObject cellPrefabWhite;
    public GameObject cellPrefabBlack;

    private GridLayoutGroup gridLayout;

    // Количество строк и столбцов
    private int rowCount = 8;
    private int columnCount = 12;

    System.Random random = new System.Random();

    public static bool targetIsHit = false;

    // Start is called before the first frame update
    void Start()
    {
        gridLayout = GetComponent<GridLayoutGroup>();
        GenerateChessboard();
    }

    // Update is called once per frame
    void Update()
    {
        if (targetIsHit)
        {
            Debug.Log("------------------------");
            targetIsHit = false;
            foreach (Transform child in transform)
            {
                Destroy(child.gameObject);
            }
            LayoutRebuilder.ForceRebuildLayoutImmediate(gridLayout.GetComponent<RectTransform>());
            GenerateChessboard();
        }
        else
        {
            return;
        }
    }

    private void GenerateChessboard()
    {
        // Генерируем координаты произвольной точки в заданном диапазоне
        int XfirstPointRandomIndex = random.Next(2, columnCount - 2);
        int YfirstPointRandomIndex = random.Next(2, rowCount - 2);

        int minX = 0;
        int maxX = 0;
        int minY = 0;
        int maxY = 0;

        int randomNumber = random.Next(1, 5);

        Debug.Log("randomLumber=" + randomNumber);

        switch (randomNumber)
        {
            case 1:
                // Точка по диагонали слева сверху от исходной
                minX = XfirstPointRandomIndex - 1;
                maxX = XfirstPointRandomIndex;
                minY = YfirstPointRandomIndex - 1;
                maxY = YfirstPointRandomIndex;
                break;
            case 2:
                // Точка по диагонали справа сверху от исходной
                minX = XfirstPointRandomIndex;
                maxX = XfirstPointRandomIndex + 1;
                minY = YfirstPointRandomIndex - 1;
                maxY = YfirstPointRandomIndex;
                break;
            case 3:
                // Точка по диагонали справа снизу от исходной
                minX = XfirstPointRandomIndex;
                maxX = XfirstPointRandomIndex + 1;
                minY = YfirstPointRandomIndex;
                maxY = YfirstPointRandomIndex + 1;
                break;
            case 4:
                // Точка по диагонали слева снизу от исходной
                minX = XfirstPointRandomIndex - 1;
                maxX = XfirstPointRandomIndex;
                minY = YfirstPointRandomIndex;
                maxY = YfirstPointRandomIndex + 1;
                break;
        }
        Debug.Log("Первая точка: (" + XfirstPointRandomIndex + ";" + YfirstPointRandomIndex + ")");
        Debug.Log("Итоговый mixX: " + minX + " maxX: " + maxX);
        Debug.Log("Итоговый minY: " + minY + " maxY: " + maxY);

        for (int row = 0; row < rowCount; row++)
        {
            for (int column = 0; column < columnCount; column++)
            {
                bool isBlack;
                bool isTarget = false;
                GameObject cell;
                if (row % 2 == 0)
                {
                    if (column % 2 == 0)
                    {
                        if (row >= minY && row <= maxY && column >= minX && column <= maxX)
                        {
                            // Инвертировали цвета для ячеек в центральном прямоугольнике 2х2
                            isTarget = true;
                            cell = Instantiate(cellPrefabWhite, transform);
                            isBlack = false;
                            Debug.Log("@ " +"(" + row + ";" + column + ")");
                        }
                        else
                        {
                            cell = Instantiate(cellPrefabBlack, transform);
                            isBlack = true;
                        }
                        
                    }
                    else
                    {
                        if (row >= minY && row <= maxY && column >= minX && column <= maxX)
                        {
                            // Инвертировали цвета для ячеек в центральном прямоугольнике 2х2
                            isTarget = true;
                            cell = Instantiate(cellPrefabBlack, transform);
                            isBlack = true;
                            Debug.Log("@ " + "(" + row + ";" + column + ")");
                        }
                        else
                        {
                            cell = Instantiate(cellPrefabWhite, transform);
                            isBlack = false;
                        }
                        
                    }
                }
                else
                {
                    if (column % 2 == 0)
                    {
                        if (row >= minY && row <= maxY && column >= minX && column <= maxX)
                        {
                            // Инвертировали цвета для ячеек в центральном прямоугольнике 2х2
                            isTarget = true;
                            cell = Instantiate(cellPrefabBlack, transform);
                            isBlack = true;
                            Debug.Log("@ " + "(" + row + ";" + column + ")");
                        }
                        else
                        {
                            cell = Instantiate(cellPrefabWhite, transform);
                            isBlack = false;
                        }
                        
                    }
                    else
                    {
                        if (row >= minY && row <= maxY && column >= minX && column <= maxX)
                        {
                            // Инвертировали цвета для ячеек в центральном прямоугольнике 2х2
                            isTarget = true;
                            cell = Instantiate(cellPrefabWhite, transform);
                            isBlack = false;
                            Debug.Log("@ " + "(" + row + ";" + column + ")");
                        }
                        else
                        {
                            cell = Instantiate(cellPrefabBlack, transform);
                            isBlack = true;
                        }
                        
                    }
                }

                // Верх, Низ от центрального прямоугольника 2х2:
                if ((row == minY - 1 || row == maxY + 1) && column >= minX && column <= maxX)
                {
                    isTarget = true;
                }
                // Лево, Право от центрального прямоугольника 2х2:
                if ((column == minX - 1 || column == maxX + 1) && row >= minY && row <= maxY)
                {
                    isTarget = true;
                }

                ChessCell chessCell = cell.GetComponent<ChessCell>();

                if (chessCell != null)
                {
                    chessCell.configure(row, column, isBlack, isTarget);
                }
                // Расположение клеток в соответствии с их позицией в шахматной доске
                cell.transform.localPosition = new Vector3(column, -row, 0);
            }
        }
    }
}
#include<stdio.h>
#include<string.h>

void swapChar(char *a, char *b)
{
	char temp = *a;
	*a = *b;
	*b = temp;
}

void reverseString(char s[])
{
	int length = strlen(s);
	for (int i = 0; i < length / 2; i++)
	{
		swapChar(&s[i], &s[length - 1 - i]);
	}
}

int main(void)
{
	char s[] = "1234567";
	printf("Original String: %s\n", s);
	reverseString(s);
	printf("Reversed String: %s", s);
	
	return 0;
}